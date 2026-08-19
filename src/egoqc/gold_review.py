from __future__ import annotations

import hashlib
import json
import os
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import av

from .report import write_json
from .validator import load_episode_index


SCHEMA_VERSION = "egoqc-phase-a-gold-review-v1"
AGGREGATE_ISSUES = {"bad_frame_ratio_exceeded"}


ISSUE_LABELS = {
    "bad_frame_ratio_exceeded": "坏帧比例超标",
    "instantaneous_velocity_outlier": "瞬时速度异常",
    "joint_rotation_jitter": "手指关节旋转抖动",
    "position_jitter": "手腕位置抖动",
    "temporal_spike": "单帧轨迹跳点",
    "wrist_rotation_jitter": "手腕旋转抖动",
    "camera_jitter": "相机位姿抖动",
    "mask_flicker": "手部有效标记闪烁",
    "pose_freeze_candidate": "手部姿态冻结",
    "beta_drift": "MANO 手型参数漂移",
}


GOLD_LABELS = [
    {"code": "hand_absent", "label": "手离画超过标准"},
    {"code": "persistent_extra_hands", "label": "出现第二个人的手"},
    {"code": "semantic_camera_shake", "label": "无意义相机抖动"},
    {"code": "severe_occlusion", "label": "严重遮挡"},
    {"code": "mano_overlay_drift", "label": "MANO mesh / 骨骼偏离真实手"},
    {"code": "task_label_mismatch", "label": "任务文本与视频不符"},
    {"code": "subtask_boundary_error", "label": "子任务边界错误"},
    {"code": "unusable_visual_quality", "label": "模糊或曝光导致不可用"},
    {"code": "action_not_observable", "label": "关键动作不可观察"},
    {"code": "interaction_incomplete", "label": "任务未完成或片段不完整"},
    {"code": "frozen_or_duplicate_frames", "label": "冻结帧或重复帧"},
    {"code": "severe_lens_artifact", "label": "严重畸变或镜头伪影"},
    {"code": "scene_task_out_of_scope", "label": "场景或任务不在需求范围"},
]


CAUSE_OPTIONS = [
    {"code": "true_fast_motion", "label": "真实快速动作"},
    {"code": "hand_tracking_drift", "label": "手部追踪缓慢漂移"},
    {"code": "hand_tracking_jump", "label": "手部追踪瞬时跳变"},
    {"code": "camera_motion", "label": "相机真实运动或 SLAM 发散"},
    {"code": "occlusion", "label": "遮挡导致证据不足"},
    {"code": "out_of_frame", "label": "手离开画面"},
    {"code": "left_right_swap", "label": "左右手交换"},
    {"code": "depth_or_scale", "label": "深度或尺度不一致"},
    {"code": "pose_flip", "label": "姿态翻转 / 180° 歧义"},
    {"code": "blur_or_exposure", "label": "模糊或曝光问题"},
    {"code": "timestamp_misalignment", "label": "视频与标注时间错位"},
    {"code": "second_person_hand", "label": "第二人手或干扰物"},
    {"code": "no_visible_problem", "label": "没有可见问题"},
    {"code": "other", "label": "其他（写在备注）"},
]


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows = []
    with path.expanduser().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} 必须是 JSON object")
            rows.append(value)
    return rows


def _fps(info: Dict[str, Any], video_key: str) -> float:
    if "fps" in info:
        return float(info["fps"])
    return float(info["features"][video_key]["info"]["video.fps"])


def _inside(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath((str(path), str(root))) == str(root)
    except ValueError:
        return False


def _route(dataset: Path, row: Dict[str, Any], video_key: str) -> Path:
    return (
        dataset
        / "videos"
        / video_key
        / f"chunk-{int(row[f'videos/{video_key}/chunk_index']):03d}"
        / f"file-{int(row[f'videos/{video_key}/file_index']):03d}.mp4"
    )


def _source_signature(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def materialize_episode_clip(
    source: Path,
    output: Path,
    *,
    start_s: float,
    frame_count: int,
    fps: float,
) -> Dict[str, Any]:
    """Create an exact derived episode MP4; never writes beside the source."""

    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance_path = output.with_suffix(".provenance.json")
    signature = _source_signature(source)
    expected = {
        "source": str(source),
        "source_signature": signature,
        "source_start_s": float(start_s),
        "frames": int(frame_count),
        "fps": float(fps),
    }
    if output.is_file() and provenance_path.is_file():
        previous = json.loads(provenance_path.read_text(encoding="utf-8"))
        if all(previous.get(key) == value for key, value in expected.items()):
            return {**previous, "cached": True}

    rate = Fraction(str(fps)).limit_denominator(1001)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.mp4")
    rendered = 0
    seek_used = False
    with av.open(str(source)) as source_container, av.open(str(temporary), mode="w") as sink:
        source_stream = source_container.streams.video[0]
        source_stream.thread_type = "AUTO"
        width = int(source_stream.codec_context.width)
        height = int(source_stream.codec_context.height)
        target = sink.add_stream("libx264", rate=rate)
        target.width = width
        target.height = height
        target.pix_fmt = "yuv420p"
        target.options = {"crf": "20", "preset": "veryfast"}
        if source_stream.time_base is not None and start_s > 0:
            offset = max(0, int((start_s - 1.0 / fps) / float(source_stream.time_base)))
            try:
                source_container.seek(offset, stream=source_stream, backward=True, any_frame=False)
                seek_used = True
            except (av.error.FFmpegError, OSError, ValueError):
                seek_used = False
        for decoded_index, frame in enumerate(source_container.decode(source_stream)):
            frame_time = float(frame.time) if frame.time is not None else None
            if frame_time is not None:
                if frame_time + 0.5 / fps < start_s:
                    continue
            elif not seek_used and decoded_index < int(round(start_s * fps)):
                continue
            if rendered >= frame_count:
                break
            output_frame = frame.reformat(width=width, height=height, format="yuv420p")
            output_frame.pts = rendered
            output_frame.time_base = Fraction(rate.denominator, rate.numerator)
            for packet in target.encode(output_frame):
                sink.mux(packet)
            rendered += 1
        for packet in target.encode():
            sink.mux(packet)
    if rendered != frame_count:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"源视频只解码出 {rendered}/{frame_count} 个 episode 帧: {source}")
    temporary.replace(output)
    provenance = {
        **expected,
        "output": str(output),
        "source_readonly": True,
        "derived_media": True,
        "seek_used": seek_used,
        "width": width,
        "height": height,
        "codec": "h264",
        "pix_fmt": "yuv420p",
        "cached": False,
    }
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return provenance


def _stable_event_id(dataset_id: str, episode_index: int) -> str:
    digest = hashlib.sha256(
        f"{SCHEMA_VERSION}:{dataset_id}:{episode_index}".encode("utf-8")
    ).hexdigest()[:16]
    return f"phase-a--episode-{episode_index:06d}--{digest}"


def _issue_labels(codes: Iterable[str]) -> Dict[str, str]:
    return {str(code): ISSUE_LABELS.get(str(code), str(code)) for code in codes}


def build_phase_a_review_events(
    dataset: Path,
    baseline_evidence: Path,
    output: Path,
    *,
    annotated_root: Optional[Path] = None,
    video_key: str = "observation.images.ego",
    materialize_media: bool = True,
) -> Dict[str, Any]:
    """Build episode-level Gold tasks for the existing PostgreSQL review panel."""

    dataset = dataset.expanduser().resolve()
    output = output.expanduser().resolve()
    if _inside(output, dataset):
        raise ValueError("Gold review 输出不能位于原始 dataset 内部")
    output.mkdir(parents=True, exist_ok=True)
    media_root = output / "media"
    media_root.mkdir(exist_ok=True)
    annotated_root = annotated_root.expanduser().resolve() if annotated_root else None
    info = json.loads((dataset / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = _fps(info, video_key)
    episode_rows = {
        int(row["episode_index"]): row for row in load_episode_index(dataset).to_pylist()
    }
    baselines = _read_jsonl(baseline_evidence)
    events = []
    raw_clips = 0
    annotated_clips = 0
    for baseline in sorted(baselines, key=lambda row: int(row["episode_index"])):
        episode_index = int(baseline["episode_index"])
        route = episode_rows.get(episode_index)
        if route is None:
            raise ValueError(f"baseline episode 不存在于 dataset: {episode_index}")
        length = int(route["length"])
        if int(baseline.get("length", length)) != length:
            raise ValueError(f"episode {episode_index} baseline length 与 metadata 不一致")
        source_video = _route(dataset, route, video_key).resolve()
        if not source_video.is_file():
            raise FileNotFoundError(source_video)
        raw_clip = media_root / f"episode-{episode_index:06d}-raw.mp4"
        if materialize_media:
            materialize_episode_clip(
                source_video,
                raw_clip,
                start_s=float(route[f"videos/{video_key}/from_timestamp"]),
                frame_count=length,
                fps=fps,
            )
            raw_clips += 1
        else:
            raw_clip = source_video

        annotated_clip = None
        if annotated_root:
            candidate = annotated_root / f"episode-{episode_index:06d}-annotated.mp4"
            if candidate.is_file():
                annotated_clip = candidate
                annotated_clips += 1

        all_issue_codes = [str(code) for code in baseline.get("issue_codes", [])]
        issue_codes = [code for code in all_issue_codes if code not in AGGREGATE_ISSUES]
        aggregate_issue_codes = [
            code for code in all_issue_codes if code in AGGREGATE_ISSUES
        ]
        duration = length / fps
        dataset_id = str(baseline.get("dataset_id") or dataset.name)
        task = "；".join(str(value) for value in baseline.get("tasks", []) if value)
        metrics = {
            "review_mode": "episode_gold",
            "schema_version": SCHEMA_VERSION,
            "episode_index": episode_index,
            "task": task,
            "baseline_tier": baseline.get("tier"),
            "issue_codes": issue_codes,
            "issue_labels": _issue_labels(issue_codes),
            "aggregate_issue_codes": aggregate_issue_codes,
            "rule_evidence": baseline.get("evidence") or {},
            "bad_frames": baseline.get("bad_frames") or [],
            "sample_frames": baseline.get("sample_frames") or [],
            "gold_labels": GOLD_LABELS,
            "cause_options": CAUSE_OPTIONS,
            "raw_clip_path": str(raw_clip),
            "annotated_clip_path": str(annotated_clip) if annotated_clip else None,
            "mano_overlay_available": annotated_clip is not None,
            "source_episode_start_s": float(
                route[f"videos/{video_key}/from_timestamp"]
            ),
            "source_episode_end_s": float(route[f"videos/{video_key}/to_timestamp"]),
            "source_data_file": baseline.get("source_data_file"),
            "source_dataset_id": dataset_id,
            "synthetic": False,
            "raw_source_readonly": True,
        }
        events.append({
            "event_id": _stable_event_id(dataset_id, episode_index),
            "video_id": f"{dataset.name}/episode-{episode_index:06d}",
            "kind": "episode_qc_gold_review",
            "category": "human_gold",
            "severity": "review",
            "start_s": 0.0,
            "end_s": duration,
            "duration_s": duration,
            "clip_path": str(raw_clip),
            "source_uri": str(source_video),
            "priority": 100 if baseline.get("tier") == "bronze" else 50,
            **metrics,
        })

    events_path = output / "review-events.json"
    events_path.write_text(
        json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset": str(dataset),
        "baseline_evidence": str(baseline_evidence.expanduser().resolve()),
        "events": len(events),
        "raw_clips": raw_clips,
        "annotated_clips": annotated_clips,
        "materialize_media": materialize_media,
        "raw_source_modified": False,
        "artifacts": {
            "events": str(events_path),
            "media_root": str(media_root),
        },
    }
    write_json(output / "summary.json", summary)
    return summary
