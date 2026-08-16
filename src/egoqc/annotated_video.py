from __future__ import annotations

import json
import os
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

from .mano import ManoOverlayRenderer
from .validator import load_episode_index


FRAME_COLUMNS = [
    "episode_index",
    "frame_index",
    "main_type",
    "state_mask",
    "observation.state",
    "fov",
    "intrinsics",
    "extrinsics_w2c",
    "left_transl_world",
    "left_orient_world",
    "left_hand_pose",
    "right_transl_world",
    "right_orient_world",
    "right_hand_pose",
]


def _fps(info: Dict[str, Any], video_key: str) -> float:
    if "fps" in info:
        return float(info["fps"])
    return float(info["features"][video_key]["info"]["video.fps"])


def _episode_route(dataset: Path, episode: int, video_key: str) -> Dict[str, Any]:
    rows = load_episode_index(dataset).to_pylist()
    matches = [row for row in rows if int(row["episode_index"]) == episode]
    if len(matches) != 1:
        raise ValueError(f"episode {episode} 路由数量应为 1，实际 {len(matches)}")
    row = matches[0]
    video_path = (
        dataset
        / "videos"
        / video_key
        / f"chunk-{int(row[f'videos/{video_key}/chunk_index']):03d}"
        / f"file-{int(row[f'videos/{video_key}/file_index']):03d}.mp4"
    )
    data_path = (
        dataset
        / "data"
        / f"chunk-{int(row['data/chunk_index']):03d}"
        / f"file-{int(row['data/file_index']):03d}.parquet"
    )
    return {
        "metadata": row,
        "video_path": video_path,
        "data_path": data_path,
        "start_time": float(row[f"videos/{video_key}/from_timestamp"]),
        "stop_time": float(row[f"videos/{video_key}/to_timestamp"]),
        "length": int(row["length"]),
    }


def _episode_records(data_path: Path, episode: int) -> List[Dict[str, Any]]:
    parquet = pq.ParquetFile(data_path)
    available = set(parquet.schema_arrow.names)
    required = set(FRAME_COLUMNS) - {"intrinsics", "main_type"}
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"annotated video 缺少字段 {missing}: {data_path}")
    table = parquet.read(columns=[name for name in FRAME_COLUMNS if name in available])
    values = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
    table = table.filter(pa.array(values == episode))
    records = table.to_pylist()
    records.sort(key=lambda row: int(row["frame_index"]))
    return records


def _load_review(path: Optional[Path], episode: int) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    for line in path.expanduser().read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if int(row["episode_index"]) == episode:
                return row
    return None


def _hud(
    image: Image.Image,
    episode: int,
    record: Dict[str, Any],
    fps: float,
    metrics: Dict[str, Any],
    review: Optional[Dict[str, Any]],
) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    frame_index = int(record["frame_index"])
    mask = np.asarray(record["state_mask"], dtype=bool).reshape(2)
    main_type = int(record.get("main_type", -1))
    hand_text = f"L={'OK' if mask[0] else 'MISS'}  R={'OK' if mask[1] else 'MISS'}"
    lines = [
        f"EgoQC  episode={episode}  frame={frame_index}  t={frame_index / fps:.3f}s",
        f"{hand_text}  main={main_type}  MANO={metrics['hands_rendered']} hand(s)",
    ]
    if review:
        lines.append(
            f"HUMAN={str(review.get('decision', 'unsure')).upper()}  {review.get('note', '')}"
        )
    height = 18 * len(lines) + 16
    draw.rounded_rectangle((12, 12, min(image.width - 12, 720), 12 + height), radius=8, fill=(8, 12, 10, 190))
    for index, line in enumerate(lines):
        draw.text((24, 21 + index * 18), line, fill=(255, 255, 255, 245))
    draw.text(
        (24, image.height - 28),
        "cyan=left mesh+skeleton  orange=right mesh+skeleton  white=joints",
        fill=(255, 255, 255, 230),
        stroke_width=2,
        stroke_fill=(0, 0, 0, 180),
    )


def render_annotated_episode(
    dataset: Path,
    episode: int,
    output: Path,
    mano_renderer: ManoOverlayRenderer,
    video_key: str = "observation.images.ego",
    batch_size: int = 32,
    start_frame: int = 0,
    max_frames: Optional[int] = None,
    review_labels: Optional[Path] = None,
    records_override: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    dataset = dataset.expanduser().resolve()
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    info = json.loads((dataset / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = _fps(info, video_key)
    rate = Fraction(str(fps)).limit_denominator(1001)
    route = _episode_route(dataset, episode, video_key)
    records = records_override if records_override is not None else _episode_records(route["data_path"], episode)
    if len(records) != route["length"]:
        raise ValueError(
            f"episode {episode} metadata length={route['length']}，Parquet rows={len(records)}"
        )
    if start_frame < 0 or start_frame >= len(records):
        raise ValueError(f"start_frame 超出范围: {start_frame}, episode length={len(records)}")
    available_count = len(records) - start_frame
    target_count = min(available_count, max_frames) if max_frames else available_count
    review = _load_review(review_labels, episode)
    target_source_time = route["start_time"] + start_frame / fps
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.mp4")
    rendered_count = 0
    hands_rendered = 0

    with av.open(str(route["video_path"])) as source, av.open(str(temporary), mode="w") as sink:
        source_stream = source.streams.video[0]
        source_stream.thread_type = "AUTO"
        width = int(source_stream.codec_context.width)
        height = int(source_stream.codec_context.height)
        target_stream = sink.add_stream("libx264", rate=rate)
        target_stream.width = width
        target_stream.height = height
        target_stream.pix_fmt = "yuv420p"
        target_stream.options = {"crf": "18", "preset": "veryfast"}
        seek_used = False
        if source_stream.time_base is not None and target_source_time > 0:
            seek_offset = max(
                0,
                int((target_source_time - 1.0 / fps) / float(source_stream.time_base)),
            )
            try:
                source.seek(seek_offset, stream=source_stream, backward=True, any_frame=False)
                seek_used = True
            except (av.error.FFmpegError, OSError, ValueError):
                seek_used = False
        pending_images: List[Image.Image] = []
        pending_records: List[Dict[str, Any]] = []
        selected_count = 0

        def flush() -> None:
            nonlocal rendered_count, hands_rendered
            if not pending_images:
                return
            overlays, metrics = mano_renderer.render_many(pending_images, pending_records)
            for image, record, frame_metrics in zip(overlays, pending_records, metrics):
                _hud(image, episode, record, fps, frame_metrics, review)
                video_frame = av.VideoFrame.from_image(image)
                video_frame.pts = rendered_count
                for packet in target_stream.encode(video_frame):
                    sink.mux(packet)
                rendered_count += 1
                hands_rendered += int(frame_metrics["hands_rendered"])
                image.close()
            for image in pending_images:
                image.close()
            pending_images.clear()
            pending_records.clear()

        for decoded_index, frame in enumerate(source.decode(source_stream)):
            frame_time = float(frame.time) if frame.time is not None else None
            if frame_time is not None:
                if frame_time + 0.5 / fps < target_source_time:
                    continue
            elif not seek_used and decoded_index < int(round(target_source_time * fps)):
                continue
            if selected_count >= target_count:
                break
            record_index = start_frame + selected_count
            record = records[record_index]
            if int(record["frame_index"]) != record_index:
                raise ValueError(f"frame_index 不连续: expected={record_index}")
            pending_images.append(frame.to_image())
            pending_records.append(record)
            selected_count += 1
            if len(pending_images) >= batch_size:
                flush()
        flush()
        for packet in target_stream.encode():
            sink.mux(packet)
    if rendered_count != target_count:
        raise ValueError(f"视频只解码出 {rendered_count}/{target_count} 个目标帧")
    temporary.replace(output)
    provenance = {
        "dataset": str(dataset),
        "episode_index": episode,
        "source_video": str(route["video_path"]),
        "source_start_time": route["start_time"],
        "source_stop_time": route["stop_time"],
        "render_start_frame": start_frame,
        "render_end_frame": start_frame + rendered_count,
        "render_start_time": route["start_time"] + start_frame / fps,
        "render_stop_time": route["start_time"] + (start_frame + rendered_count) / fps,
        "source_seek_used": seek_used,
        "output": str(output),
        "frames": rendered_count,
        "fps": fps,
        "width": width,
        "height": height,
        "codec": "h264",
        "pix_fmt": "yuv420p",
        "hands_rendered": hands_rendered,
        "review_label_embedded": review is not None,
        "record_source": "override" if records_override is not None else "source_parquet",
        "mano": mano_renderer.provenance(),
    }
    provenance_path = output.with_suffix(".provenance.json")
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    return provenance
