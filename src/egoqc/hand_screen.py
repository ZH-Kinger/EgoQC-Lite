from __future__ import annotations

import bisect
import concurrent.futures
import json
import multiprocessing
import tarfile
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .adapters import RekaDailyRawAdapter
from .report import write_json


def _segments(mask: Sequence[bool], times: Sequence[float], value: bool, dt: float) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    start: Optional[int] = None
    for index, item in enumerate(mask):
        if item == value and start is None:
            start = index
        if item != value and start is not None:
            segments.append({
                "start_sample": start,
                "end_sample": index - 1,
                "start_s": float(times[start]),
                "end_s": float(times[index - 1] + dt),
                "duration_s": float(times[index - 1] + dt - times[start]),
            })
            start = None
    if start is not None and times:
        segments.append({
            "start_sample": start,
            "end_sample": len(mask) - 1,
            "start_s": float(times[start]),
            "end_s": float(times[-1] + dt),
            "duration_s": float(times[-1] + dt - times[start]),
        })
    return segments


def _bridge_short_gaps(mask: Sequence[bool], times: Sequence[float], dt: float, maximum_s: float) -> List[bool]:
    result = list(mask)
    for segment in _segments(mask, times, False, dt):
        start = int(segment["start_sample"])
        end = int(segment["end_sample"])
        internal = start > 0 and end + 1 < len(result) and result[start - 1] and result[end + 1]
        if internal and float(segment["duration_s"]) <= maximum_s + 1e-9:
            result[start : end + 1] = [True] * (end - start + 1)
    return result


def _box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _robust_hand_count(
    sample: Dict[str, Any], confidence: float, nms_iou: float
) -> int:
    detections = sorted(
        (
            detection for detection in sample.get("detections", [])
            if float(detection[4]) >= confidence
        ),
        key=lambda detection: float(detection[4]),
        reverse=True,
    )
    kept: List[Sequence[float]] = []
    for detection in detections:
        if all(_box_iou(detection, existing) < nms_iou for existing in kept):
            kept.append(detection)
    return len(kept)


def summarize_hand_samples(
    samples: Sequence[Dict[str, Any]],
    sample_fps: float,
    bridge_gap_s: float = 0.4,
    absence_limit_s: float = 1.0,
    minimum_run_s: float = 5.0,
    extra_hand_confidence: float = 0.7,
    extra_hand_nms_iou: float = 0.5,
    extra_hand_persistence_s: float = 0.6,
) -> Dict[str, Any]:
    dt = 1.0 / sample_fps
    times = [float(sample["time_s"]) for sample in samples]
    raw_presence = [int(sample["hand_count"]) > 0 for sample in samples]
    presence = _bridge_short_gaps(raw_presence, times, dt, bridge_gap_s)
    visible_runs = _segments(presence, times, True, dt)
    absent_runs = _segments(presence, times, False, dt)
    long_absences = [segment for segment in absent_runs if segment["duration_s"] > absence_limit_s]
    valid_duration = sum(
        max(0.0, float(segment["duration_s"]) - minimum_run_s)
        for segment in visible_runs
        if float(segment["duration_s"]) > minimum_run_s
    )
    count = max(1, len(samples))
    visible_ratio = sum(presence) / count
    both_ratio = sum(int(sample["hand_count"]) >= 2 for sample in samples) / count
    raw_extra_ratio = sum(int(sample["hand_count"]) > 2 for sample in samples) / count
    robust_extra_mask = [
        _robust_hand_count(sample, extra_hand_confidence, extra_hand_nms_iou) > 2
        for sample in samples
    ]
    robust_extra_runs = _segments(robust_extra_mask, times, True, dt)
    persistent_extra_runs = [
        segment for segment in robust_extra_runs
        if float(segment["duration_s"]) >= extra_hand_persistence_s
    ]
    extra_ratio = sum(robust_extra_mask) / count
    edge_ratio = sum(bool(sample.get("edge_touch")) for sample in samples) / count
    mean_confidence = float(np.mean([
        float(detection[4])
        for sample in samples for detection in sample.get("detections", [])
    ])) if any(sample.get("detections") for sample in samples) else None
    if visible_ratio < 0.2 or valid_duration <= 0:
        decision = "screen_out_before_mano"
    elif long_absences or persistent_extra_runs:
        decision = "review_before_mano"
    else:
        decision = "candidate_for_mano"
    return {
        "sample_count": len(samples),
        "sample_fps": sample_fps,
        "bridge_gap_s": bridge_gap_s,
        "absence_limit_s": absence_limit_s,
        "minimum_visible_run_s": minimum_run_s,
        "any_hand_visible_ratio": visible_ratio,
        "both_hands_visible_ratio": both_ratio,
        "raw_extra_hands_ratio": raw_extra_ratio,
        "suspected_extra_hands_ratio": extra_ratio,
        "extra_hand_confidence": extra_hand_confidence,
        "extra_hand_nms_iou": extra_hand_nms_iou,
        "extra_hand_persistence_s": extra_hand_persistence_s,
        "suspected_extra_hand_segments": persistent_extra_runs,
        "edge_touch_ratio": edge_ratio,
        "mean_detection_confidence": mean_confidence,
        "longest_no_hand_gap_s": max((float(segment["duration_s"]) for segment in absent_runs), default=0.0),
        "long_no_hand_segments": long_absences,
        "visible_segments": visible_runs,
        "effective_video_duration_s": valid_duration,
        "provisional_decision": decision,
        "decision_semantics": "model_screening_only; procurement decision requires human review",
    }


@contextmanager
def _open_video(dataset: Path, row: Dict[str, Any]) -> Iterator[Any]:
    import av

    adapter = RekaDailyRawAdapter()
    with ExitStack() as stack:
        tar_video = adapter._tar_video(dataset, row)
        if tar_video:
            shard, member = tar_video
            archive = stack.enter_context(tarfile.open(shard, "r"))
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"tar 成员不是普通文件: {member}")
            yield stack.enter_context(av.open(stream)), f"tar://{shard}!/{member}"
            return
        loose = adapter._loose_video(dataset, row)
        if loose and loose.suffix.lower() in {".mp4", ".mov", ".avi"}:
            yield stack.enter_context(av.open(str(loose))), str(loose)
            return
        raise FileNotFoundError(str(row["video_id"]))


def _result_samples(results: Iterable[Any], times: Sequence[float], shapes: Sequence[Tuple[int, int]]) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for result, timestamp, (height, width) in zip(results, times, shapes):
        detections: List[List[float]] = []
        edge_touch = False
        if result.boxes is not None:
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            confidences = result.boxes.conf.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy()
            margin_x, margin_y = 0.02 * width, 0.02 * height
            for box, confidence, hand_class in zip(boxes, confidences, classes):
                x1, y1, x2, y2 = [float(value) for value in box]
                detections.append([x1, y1, x2, y2, float(confidence), int(hand_class)])
                edge_touch = edge_touch or x1 <= margin_x or y1 <= margin_y or x2 >= width - margin_x or y2 >= height - margin_y
        samples.append({
            "time_s": float(timestamp),
            "hand_count": len(detections),
            "left_present": any(int(detection[5]) == 0 for detection in detections),
            "right_present": any(int(detection[5]) == 1 for detection in detections),
            "edge_touch": edge_touch,
            "detections": detections,
        })
    return samples


def detect_video_hands(
    model: Any,
    dataset: Path,
    row: Dict[str, Any],
    sample_fps: float,
    confidence: float,
    batch_size: int,
    device: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    started = time.monotonic()
    samples: List[Dict[str, Any]] = []
    images: List[np.ndarray] = []
    times: List[float] = []
    shapes: List[Tuple[int, int]] = []
    decoded_frames = 0
    next_time = 0.0
    source_uri = None

    def infer_pending() -> None:
        if not images:
            return
        results = model.predict(
            images,
            conf=confidence,
            device=device,
            verbose=False,
        )
        samples.extend(_result_samples(results, times, shapes))
        images.clear(); times.clear(); shapes.clear()

    with _open_video(dataset, row) as (container, source):
        source_uri = source
        stream = container.streams.video[0]
        rate = float(stream.average_rate) if stream.average_rate else float(row.get("fps") or 30.0)
        for frame in container.decode(stream):
            timestamp = float(frame.time) if frame.time is not None else decoded_frames / rate
            decoded_frames += 1
            if timestamp + 0.5 / rate < next_time:
                continue
            image = frame.to_ndarray(format="bgr24")
            images.append(image)
            times.append(timestamp)
            shapes.append((image.shape[0], image.shape[1]))
            next_time += 1.0 / sample_fps
            if len(images) >= batch_size:
                infer_pending()
        infer_pending()
    return samples, {
        "source_uri": source_uri,
        "decoded_frames": decoded_frames,
        "sampled_frames": len(samples),
        "wall_seconds": time.monotonic() - started,
    }


def _font(size: int) -> ImageFont.ImageFont:
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _evidence_times(samples: Sequence[Dict[str, Any]], metrics: Dict[str, Any], duration: float) -> List[float]:
    candidates = list(np.linspace(0, max(0.0, duration - 0.01), 6))
    for segment in sorted(metrics["long_no_hand_segments"], key=lambda item: item["duration_s"], reverse=True)[:3]:
        candidates.append((float(segment["start_s"]) + float(segment["end_s"])) / 2)
    for segment in sorted(
        metrics.get("suspected_extra_hand_segments", []),
        key=lambda item: item["duration_s"],
        reverse=True,
    )[:3]:
        candidates.append((float(segment["start_s"]) + float(segment["end_s"])) / 2)
    return sorted({round(max(0.0, value), 3) for value in candidates})[:12]


def render_hand_evidence(
    dataset: Path, row: Dict[str, Any], samples: Sequence[Dict[str, Any]], metrics: Dict[str, Any], output: Path
) -> None:
    import av

    duration = float(row.get("duration_s") or 0)
    if not samples:
        return
    targets = _evidence_times(samples, metrics, duration)
    cell_w, cell_h, gutter, header = 480, 270, 8, 46
    rows = max(1, (len(targets) + 2) // 3)
    canvas = Image.new("RGB", (cell_w * 3 + gutter * 4, header + rows * (cell_h + gutter) + gutter), "#ecece7")
    draw = ImageDraw.Draw(canvas)
    draw.text((gutter, 10), str(row["video_id"]), fill="#11110f", font=_font(19))
    draw.text((canvas.width - gutter, 14), metrics["provisional_decision"].upper(), fill="#b34824", font=_font(12), anchor="ra")
    sample_times = [float(sample["time_s"]) for sample in samples]
    with _open_video(dataset, row) as (container, _):
        stream = container.streams.video[0]
        for index, target in enumerate(targets):
            container.seek(max(0, int(target * av.time_base)), any_frame=False, backward=True)
            frame = next((item for item in container.decode(stream) if item.time is None or float(item.time) >= target - 0.05), None)
            if frame is None:
                continue
            image = Image.fromarray(frame.to_ndarray(format="rgb24"))
            scale = min(cell_w / image.width, cell_h / image.height)
            resized = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.Resampling.LANCZOS,
            )
            fitted = Image.new("RGB", (cell_w, cell_h), "#11110f")
            offset_x = (cell_w - resized.width) // 2
            offset_y = (cell_h - resized.height) // 2
            fitted.paste(resized, (offset_x, offset_y))
            nearest = min(max(0, bisect.bisect_left(sample_times, target)), len(samples) - 1)
            if nearest and abs(sample_times[nearest - 1] - target) < abs(sample_times[nearest] - target):
                nearest -= 1
            frame_draw = ImageDraw.Draw(fitted)
            for x1, y1, x2, y2, confidence, hand_class in samples[nearest]["detections"]:
                color = "#17806d" if int(hand_class) == 0 else "#b34824"
                transformed = (
                    x1 * scale + offset_x, y1 * scale + offset_y,
                    x2 * scale + offset_x, y2 * scale + offset_y,
                )
                frame_draw.rectangle(transformed, outline=color, width=4)
                frame_draw.text((transformed[0] + 4, transformed[1] + 3), f"{'L' if int(hand_class)==0 else 'R'} {confidence:.2f}", fill="white", stroke_fill="#11110f", stroke_width=2, font=_font(13))
            x = gutter + (index % 3) * (cell_w + gutter)
            y = header + gutter + (index // 3) * (cell_h + gutter)
            canvas.paste(fitted, (x, y))
            draw.rectangle((x, y + cell_h - 25, x + 92, y + cell_h), fill="#11110f")
            draw.text((x + 7, y + cell_h - 20), f"{target:.1f}s · {samples[nearest]['hand_count']}H", fill="white", font=_font(12))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=88, optimize=True)


def _screen_video_group(job: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        from ultralytics import YOLO
        import ultralytics
        import torch
    except ImportError as exc:
        raise RuntimeError("screen-rekadaily-hands 需要可选 ultralytics 环境") from exc
    dataset = Path(job["dataset"])
    output = Path(job["output"])
    weights = Path(job["weights"])
    model = YOLO(str(weights))
    adapter = RekaDailyRawAdapter()
    reports: List[Dict[str, Any]] = []
    for video_id in job["video_ids"]:
        report_path = output / video_id / "hand-screen.json"
        if job["resume"] and report_path.exists():
            reports.append(json.loads(report_path.read_text(encoding="utf-8")))
            continue
        row = adapter._row(dataset, video_id)
        samples, runtime = detect_video_hands(
            model,
            dataset,
            row,
            job["sample_fps"],
            job["confidence"],
            job["batch_size"],
            job["device"],
        )
        metrics = summarize_hand_samples(samples, job["sample_fps"])
        report = {
            "video_id": video_id,
            "dataset": str(dataset.resolve()),
            "metadata": row,
            "detector": {
                "backend": "hawor_wilor_yolo",
                "weights": str(weights.resolve()),
                "confidence": job["confidence"],
                "device": job["device"],
                "ultralytics_version": ultralytics.__version__,
                "torch_version": torch.__version__,
                "research_license_review_required": True,
            },
            "runtime": runtime,
            "metrics": metrics,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(report_path, report)
        with (report_path.parent / "hand-samples.jsonl").open("w", encoding="utf-8") as stream:
            for sample in samples:
                stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
        render_hand_evidence(
            dataset, row, samples, metrics, report_path.parent / "hand-evidence.jpg"
        )
        reports.append(report)
    return reports


def _balanced_groups(
    adapter: RekaDailyRawAdapter, dataset: Path, video_ids: Sequence[str], workers: int
) -> List[List[str]]:
    group_count = min(max(1, workers), max(1, len(video_ids)))
    groups: List[List[str]] = [[] for _ in range(group_count)]
    loads = [0.0] * group_count
    rows = [(video_id, adapter._row(dataset, video_id)) for video_id in video_ids]
    for video_id, row in sorted(
        rows, key=lambda item: float(item[1].get("duration_s") or 0), reverse=True
    ):
        target = min(range(group_count), key=loads.__getitem__)
        groups[target].append(video_id)
        loads[target] += float(row.get("duration_s") or 0)
    return [group for group in groups if group]


def screen_rekadaily_hands(
    dataset: Path,
    video_ids: Sequence[str],
    output: Path,
    weights: Path,
    sample_fps: float = 5.0,
    confidence: float = 0.2,
    batch_size: int = 32,
    device: str = "0",
    resume: bool = True,
    workers: int = 1,
) -> Dict[str, Any]:
    if workers < 1:
        raise ValueError("workers 必须 >= 1")
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    adapter = RekaDailyRawAdapter()
    reports: List[Dict[str, Any]] = []
    groups = _balanced_groups(adapter, dataset, video_ids, workers)
    common = {
        "dataset": str(dataset), "output": str(output), "weights": str(weights),
        "sample_fps": sample_fps, "confidence": confidence, "batch_size": batch_size,
        "device": device, "resume": resume,
    }
    jobs = [{**common, "video_ids": group} for group in groups]
    if len(jobs) == 1:
        reports.extend(_screen_video_group(jobs[0]))
    else:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(jobs), mp_context=context
        ) as executor:
            for group_reports in executor.map(_screen_video_group, jobs):
                reports.extend(group_reports)
    reports.sort(key=lambda report: report["video_id"])
    summary = {
        "dataset": str(dataset.resolve()),
        "videos": len(reports),
        "sample_fps": sample_fps,
        "decision_counts": {
            decision: sum(report["metrics"]["provisional_decision"] == decision for report in reports)
            for decision in ("candidate_for_mano", "review_before_mano", "screen_out_before_mano")
        },
        "effective_video_duration_s": sum(report["metrics"]["effective_video_duration_s"] for report in reports),
        "workers": len(groups),
        "wall_seconds": time.monotonic() - started,
        "aggregate_video_wall_seconds": sum(report["runtime"]["wall_seconds"] for report in reports),
        "reports": [str(output / report["video_id"] / "hand-screen.json") for report in reports],
    }
    write_json(output / "hand-screen-summary.json", summary)
    return summary
