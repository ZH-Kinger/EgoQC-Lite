from __future__ import annotations

import hashlib
import json
import tarfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .report import write_json


def _manifest_rows(
    manifest: Path,
    split: str,
    allow_technical_candidates: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with manifest.expanduser().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"manifest 第 {line_number} 行不是合法 JSON") from error
            contract = row.get("vla_pretraining", {})
            if contract.get("split") != split:
                continue
            if not contract.get("training_ready"):
                if not allow_technical_candidates or not contract.get("candidate"):
                    continue
            if not row.get("source_uri"):
                continue
            rows.append(row)
    return rows


def _apply_synthetic_augmentation(
    frames: np.ndarray, augmentation: Optional[Dict[str, Any]]
) -> np.ndarray:
    if not augmentation:
        return frames
    kind = str(augmentation.get("kind") or "")
    result = frames.copy()
    frame_count, height, width = result.shape[:3]
    seed = int(augmentation.get("seed") or 0)
    rng = np.random.default_rng(seed)
    if kind == "freeze_segment":
        start = min(frame_count - 1, max(0, int(frame_count * float(augmentation["start_fraction"]))))
        length = max(1, int(round(frame_count * float(augmentation["duration_fraction"]))))
        result[start:min(frame_count, start + length)] = result[start]
    elif kind == "blur_downsample":
        scale = float(augmentation["downsample_scale"])
        small_size = (max(8, int(width * scale)), max(8, int(height * scale)))
        radius = float(augmentation["gaussian_radius"])
        for index, frame in enumerate(result):
            image = Image.fromarray(frame).resize(small_size, Image.Resampling.BILINEAR)
            image = image.resize((width, height), Image.Resampling.BILINEAR)
            result[index] = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=radius)))
    elif kind == "foreground_occlusion":
        box_width = int(width * float(augmentation["width_fraction"]))
        box_height = int(height * float(augmentation["height_fraction"]))
        center_x = int(width * float(augmentation["center_x_fraction"]))
        center_y = int(height * float(augmentation["center_y_fraction"]))
        x0 = max(0, center_x - box_width // 2)
        y0 = max(0, center_y - box_height // 2)
        result[:, y0:min(height, y0 + box_height), x0:min(width, x0 + box_width)] = (28, 28, 25)
    elif kind == "camera_shake":
        max_dx = width * float(augmentation["max_translation_ratio"])
        max_dy = height * float(augmentation["max_translation_ratio"])
        max_angle = float(augmentation["max_rotation_deg"])
        for index, frame in enumerate(result):
            sign = -1.0 if index % 2 else 1.0
            dx = sign * rng.uniform(0.55, 1.0) * max_dx
            dy = -sign * rng.uniform(0.35, 1.0) * max_dy
            angle = sign * rng.uniform(0.45, 1.0) * max_angle
            image = Image.fromarray(frame).rotate(
                angle,
                resample=Image.Resampling.BILINEAR,
                translate=(dx, dy),
                fillcolor=(0, 0, 0),
            )
            result[index] = np.asarray(image)
    else:
        raise ValueError(f"未知 synthetic_augmentation kind={kind}")
    return result


@contextmanager
def _open_video(source_uri: str):
    if not source_uri.startswith("tar://"):
        with av.open(source_uri) as container:
            yield container
        return
    tar_path, separator, member_name = source_uri[len("tar://"):].partition("!/")
    if not separator or not tar_path or not member_name:
        raise ValueError(f"非法 tar URI: {source_uri}")
    with tarfile.open(tar_path, "r") as archive:
        member = archive.extractfile(member_name)
        if member is None:
            raise ValueError(f"tar 成员不是普通文件: {source_uri}")
        with av.open(member, metadata_errors="ignore") as container:
            yield container


def _frame_time(frame: av.VideoFrame, fallback_index: int, fallback_fps: float) -> float:
    if frame.time is not None:
        return float(frame.time)
    if frame.pts is not None and frame.time_base is not None:
        return float(frame.pts * frame.time_base)
    return fallback_index / max(fallback_fps, 1e-6)


def _center_crop_resize(array: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    target_width, target_height = size
    height, width = array.shape[:2]
    target_ratio = target_width / target_height
    source_ratio = width / height
    if source_ratio > target_ratio:
        crop_width = max(1, int(round(height * target_ratio)))
        left = (width - crop_width) // 2
        array = array[:, left:left + crop_width]
    else:
        crop_height = max(1, int(round(width / target_ratio)))
        top = (height - crop_height) // 2
        array = array[top:top + crop_height]
    image = Image.fromarray(array)
    image = image.resize((target_width, target_height), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def _letterbox_resize(array: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize without cropping edge/bottom ego hands, then pad to the target."""

    target_width, target_height = size
    image = Image.fromarray(array)
    image.thumbnail((target_width, target_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    left = (target_width - image.width) // 2
    top = (target_height - image.height) // 2
    canvas.paste(image, (left, top))
    return np.asarray(canvas, dtype=np.uint8)


class VLAPretrainDataset:
    """Framework-neutral reader for EgoQC VLA manifests.

    The returned target masks are authoritative: absent MANO/robot targets must never
    contribute to the corresponding loss.
    """

    def __init__(
        self,
        manifest: Path,
        *,
        split: str = "train",
        allow_technical_candidates: bool = False,
        seed: int = 0,
        output_size: Tuple[int, int] = (224, 224),
        resize_mode: str = "center_crop",
    ) -> None:
        self.manifest = manifest.expanduser().resolve()
        self.rows = _manifest_rows(self.manifest, split, allow_technical_candidates)
        self.split = split
        self.seed = seed
        self.epoch = 0
        self.output_size = output_size
        if resize_mode not in {"center_crop", "letterbox"}:
            raise ValueError(f"不支持 resize_mode={resize_mode}")
        self.resize_mode = resize_mode

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _start_time(self, row: Dict[str, Any]) -> float:
        sampler = row["vla_pretraining"]["clip_sampler"]
        if sampler.get("fixed_start_s") is not None:
            return float(sampler["fixed_start_s"])
        window_s = float(sampler["window_s"])
        maximum = max(0.0, float(row.get("duration_s") or 0.0) - window_s)
        identity = f"{self.seed}:{self.epoch}:{row['video_id']}"
        unit = int(hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16], 16) / float(16**16 - 1)
        return maximum * unit

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        contract = row["vla_pretraining"]
        sampler = contract["clip_sampler"]
        start_s = self._start_time(row)
        window_s = float(sampler["window_s"])
        decode_fps = float(sampler["decode_fps"])
        frame_count = max(1, int(round(window_s * decode_fps)))
        target_times = [start_s + offset / decode_fps for offset in range(frame_count)]
        decoded: List[np.ndarray] = []
        target_index = 0
        source_fps = float(row.get("fps") or decode_fps)

        with _open_video(str(row["source_uri"])) as container:
            stream = next((item for item in container.streams if item.type == "video"), None)
            if stream is None:
                raise ValueError(f"视频流不存在: {row['source_uri']}")
            if start_s > 0:
                try:
                    container.seek(int(start_s * av.time_base), backward=True, any_frame=False)
                except (av.error.FFmpegError, OSError, ValueError):
                    pass
            for decoded_index, frame in enumerate(container.decode(stream)):
                time_s = _frame_time(frame, decoded_index, source_fps)
                if time_s + 1e-6 < target_times[target_index]:
                    continue
                image = frame.to_ndarray(format="rgb24")
                resized = (
                    _letterbox_resize(image, self.output_size)
                    if self.resize_mode == "letterbox"
                    else _center_crop_resize(image, self.output_size)
                )
                while target_index < frame_count and time_s + 1e-6 >= target_times[target_index]:
                    decoded.append(resized)
                    target_index += 1
                if target_index >= frame_count:
                    break
        if not decoded:
            raise ValueError(f"无法解码目标窗口: {row['source_uri']}")
        while len(decoded) < frame_count:
            decoded.append(decoded[-1])

        activities = row.get("activities")
        if isinstance(activities, list):
            text = "; ".join(str(value) for value in activities if value)
        else:
            text = str(activities or row.get("subcategory") or row.get("category") or "")
        stacked = np.stack(decoded[:frame_count], axis=0)
        stacked = _apply_synthetic_augmentation(
            stacked, contract.get("synthetic_augmentation")
        )
        return {
            "video_id": row["video_id"],
            "frames": stacked,
            "text": text,
            "loss_masks": {key: np.float32(value) for key, value in contract["loss_masks"].items()},
            "objectives": list(contract["allowed_objectives"]),
            "source_uri": row["source_uri"],
            "clip_start_s": start_s,
            "clip_end_s": start_s + window_s,
            "provenance": row.get("provenance", {}),
            "distillation": row.get("distillation"),
        }


def collate_vla_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        raise ValueError("samples 不能为空")
    mask_names = sorted(samples[0]["loss_masks"])
    return {
        "video_ids": [sample["video_id"] for sample in samples],
        "frames": np.stack([sample["frames"] for sample in samples], axis=0),
        "texts": [sample["text"] for sample in samples],
        "loss_masks": {
            name: np.asarray([sample["loss_masks"][name] for sample in samples], dtype=np.float32)
            for name in mask_names
        },
        "clip_start_s": np.asarray([sample["clip_start_s"] for sample in samples], dtype=np.float32),
    }


def _write_contact_sheet(path: Path, samples: List[Dict[str, Any]]) -> None:
    columns = 4
    shown_per_sample = 4
    tile_width, tile_height = 224, 224
    label_height = 34
    sheet = Image.new("RGB", (columns * tile_width, len(samples) * (tile_height + label_height)), "#111410")
    draw = ImageDraw.Draw(sheet)
    for row_index, sample in enumerate(samples):
        frames = sample["frames"]
        picks = np.linspace(0, len(frames) - 1, shown_per_sample).astype(int)
        y = row_index * (tile_height + label_height)
        for column, frame_index in enumerate(picks):
            sheet.paste(Image.fromarray(frames[frame_index]), (column * tile_width, y))
        draw.text((8, y + tile_height + 8), f"{sample['video_id']} · {sample['clip_start_s']:.2f}s · {sample['text'][:80]}", fill="#e5eadf")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=88)


def smoke_vla_loader(
    manifest: Path,
    output: Path,
    *,
    split: str = "train",
    batch_size: int = 2,
    allow_technical_candidates: bool = False,
    seed: int = 0,
) -> Dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size 必须 >= 1")
    dataset = VLAPretrainDataset(
        manifest,
        split=split,
        allow_technical_candidates=allow_technical_candidates,
        seed=seed,
    )
    if not dataset:
        raise ValueError("当前筛选条件下没有可加载样本")
    samples = [dataset[index] for index in range(min(batch_size, len(dataset)))]
    batch = collate_vla_samples(samples)
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_contact_sheet(output / "vla-loader-contact-sheet.jpg", samples)
    summary = {
        "manifest": str(manifest.expanduser().resolve()),
        "split": split,
        "dataset_samples": len(dataset),
        "batch_size": len(samples),
        "frames_shape": list(batch["frames"].shape),
        "frames_dtype": str(batch["frames"].dtype),
        "video_ids": batch["video_ids"],
        "texts": batch["texts"],
        "loss_masks": {key: value.tolist() for key, value in batch["loss_masks"].items()},
        "technical_candidate_mode": allow_technical_candidates,
        "evidence": str(output / "vla-loader-contact-sheet.jpg"),
    }
    write_json(output / "vla-loader-smoke.json", summary)
    return summary
