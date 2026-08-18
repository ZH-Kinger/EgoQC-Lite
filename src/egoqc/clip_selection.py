from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .provenance import code_version
from .report import write_json, write_jsonl
from .validator import load_episode_index


SCHEMA_VERSION = "egoqc-qc-clip-candidates-v1"

EVENT_TASKS = {
    "hand_out_of_view_too_long": ("hand_absent",),
    "mask_flicker": ("mano_overlay_drift",),
    "temporal_spike": ("semantic_camera_shake", "mano_overlay_drift"),
    "instantaneous_velocity_outlier": ("semantic_camera_shake", "mano_overlay_drift"),
    "world_camera_position_mismatch": ("mano_overlay_drift",),
    "world_camera_rotation_mismatch": ("mano_overlay_drift",),
    "pose_representation_mismatch": ("mano_overlay_drift",),
    "invalid_rotation_matrix": ("mano_overlay_drift",),
    "timestamp_mismatch": (),
    "numeric_frame_interval_jitter": (),
    "numeric_non_monotonic_timestamps": (),
    "kept_mask_mismatch": ("mano_overlay_drift",),
    "non_finite_values": (),
}


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path} 第 {line_number} 行不是合法 JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path} 第 {line_number} 行必须是对象")
        rows.append(row)
    return rows


def _fps(dataset: Path, video_key: str) -> float:
    info = json.loads((dataset / "meta" / "info.json").read_text(encoding="utf-8"))
    if "fps" in info:
        return float(info["fps"])
    return float(info["features"][video_key]["info"]["video.fps"])


def _cluster_frames(
    frames: Sequence[int],
    *,
    merge_gap_frames: int,
    maximum_span_frames: int,
) -> List[List[int]]:
    clusters: List[List[int]] = []
    for frame in sorted(set(int(value) for value in frames)):
        if (
            not clusters
            or frame - clusters[-1][-1] > merge_gap_frames
            or frame - clusters[-1][0] > maximum_span_frames
        ):
            clusters.append([frame])
        else:
            clusters[-1].append(frame)
    return clusters


def _window(
    first_frame: int,
    last_frame: int,
    length: int,
    fps: float,
    *,
    minimum_s: float,
    maximum_s: float,
    context_s: float,
) -> Tuple[int, int]:
    minimum_frames = max(1, int(round(minimum_s * fps)))
    maximum_frames = max(minimum_frames, int(round(maximum_s * fps)))
    context_frames = max(0, int(round(context_s * fps)))
    start = max(0, first_frame - context_frames)
    end = min(length, last_frame + context_frames + 1)
    if end - start > maximum_frames:
        centre = (first_frame + last_frame + 1) // 2
        start = max(0, centre - maximum_frames // 2)
        end = min(length, start + maximum_frames)
        start = max(0, end - maximum_frames)
    if end - start < minimum_frames:
        desired = min(length, minimum_frames)
        centre = (start + end) // 2
        start = centre - desired // 2
        start = min(max(0, start), max(0, length - desired))
        end = start + desired
    return int(start), int(end)


def _overlaps(start: int, end: int, intervals: Iterable[Tuple[int, int]]) -> bool:
    return any(start < other_end and end > other_start for other_start, other_end in intervals)


def _clip_id(
    dataset_identity: str,
    episode_index: int,
    start_frame: int,
    end_frame: int,
    kind: str,
) -> str:
    identity = f"{dataset_identity}:{episode_index}:{start_frame}:{end_frame}:{kind}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"ep{episode_index:06d}-{start_frame:07d}-{end_frame:07d}-{digest}"


def plan_qc_clips(
    dataset: Path,
    quality_root: Path,
    output: Path,
    task_config: Path,
    *,
    video_key: str = "observation.images.ego",
    minimum_s: float = 4.0,
    maximum_s: float = 8.0,
    context_s: float = 1.5,
    merge_gap_s: float = 1.0,
    control_ratio: float = 0.25,
    minimum_control_clips: int = 8,
    maximum_clips: Optional[int] = None,
    seed: int = 17,
    source_dataset: Optional[str] = None,
    supplier_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Turn deterministic frame events into bounded visual teacher candidates."""

    dataset = dataset.expanduser().resolve()
    quality_root = quality_root.expanduser().resolve()
    output = output.expanduser().resolve()
    if minimum_s <= 0 or maximum_s < minimum_s:
        raise ValueError("clip 时长要求必须满足 0 < minimum_s <= maximum_s")
    if context_s < 0 or merge_gap_s < 0 or control_ratio < 0:
        raise ValueError("context、merge gap 和 control ratio 不能为负数")
    if minimum_control_clips < 0:
        raise ValueError("minimum control clips 不能为负数")
    tasks_config = json.loads(task_config.read_text(encoding="utf-8"))
    model_tasks = list(tasks_config["model_tasks"])
    assessment_dimensions = dict(tasks_config.get("assessment_dimensions", {}))
    fps = _fps(dataset, video_key)
    episode_results = {
        int(row["episode_index"]): row for row in _jsonl(quality_root / "episodes.jsonl")
    }
    events_by_episode: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for event in _jsonl(quality_root / "bad_frames.jsonl"):
        episode_index = int(event["episode_index"])
        frame_index = int(event["frame_index"])
        if episode_index in episode_results and 0 <= frame_index < int(
            episode_results[episode_index]["length"]
        ):
            events_by_episode[episode_index].append(event)

    route_rows = {int(row["episode_index"]): row for row in load_episode_index(dataset).to_pylist()}
    candidates: List[Dict[str, Any]] = []
    occupied: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    merge_gap_frames = max(0, int(round(merge_gap_s * fps)))
    maximum_span_frames = max(1, int(round(max(0.0, maximum_s - 2 * context_s) * fps)))

    for episode_index in sorted(events_by_episode):
        episode_result = episode_results[episode_index]
        length = int(episode_result["length"])
        route = route_rows.get(episode_index)
        if route is None:
            continue
        by_frame: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for event in events_by_episode[episode_index]:
            by_frame[int(event["frame_index"])].append(event)
        for cluster in _cluster_frames(
            list(by_frame),
            merge_gap_frames=merge_gap_frames,
            maximum_span_frames=maximum_span_frames,
        ):
            start, end = _window(
                cluster[0],
                cluster[-1],
                length,
                fps,
                minimum_s=minimum_s,
                maximum_s=maximum_s,
                context_s=context_s,
            )
            cluster_events = [event for frame in cluster for event in by_frame[frame]]
            event_codes = sorted({str(event["code"]) for event in cluster_events})
            candidate_tasks = sorted(
                {
                    task
                    for code in event_codes
                    for task in EVENT_TASKS.get(code, ())
                    if task in model_tasks
                }
            )
            candidates.append({
                "episode_index": episode_index,
                "start_frame": start,
                "end_frame": end,
                "event_frames": cluster,
                "event_codes": event_codes,
                "candidate_tasks": candidate_tasks,
                "selection_source": "deterministic_bad_frame",
                "priority": "high" if any(event.get("severity") == "error" for event in cluster_events) else "medium",
            })
            occupied[episode_index].append((start, end))

    positive_count = len(candidates)
    control_target = max(
        minimum_control_clips,
        int(round(positive_count * control_ratio)),
    )
    rng = np.random.default_rng(seed)
    episode_ids = sorted(episode_results)
    attempts = 0
    maximum_control_attempts = max(100, control_target * 50)
    minimum_frames = max(1, int(round(minimum_s * fps)))
    requested_controls = control_target
    while control_target > 0 and episode_ids and attempts < maximum_control_attempts:
        attempts += 1
        episode_index = int(rng.choice(episode_ids))
        length = int(episode_results[episode_index]["length"])
        if length < minimum_frames or episode_index not in route_rows:
            continue
        start = int(rng.integers(0, max(1, length - minimum_frames + 1)))
        end = min(length, start + minimum_frames)
        if _overlaps(start, end, occupied[episode_index]):
            continue
        candidates.append({
            "episode_index": episode_index,
            "start_frame": start,
            "end_frame": end,
            "event_frames": [],
            "event_codes": [],
            "candidate_tasks": model_tasks,
            "selection_source": "random_control_unlabeled",
            "priority": "normal",
        })
        occupied[episode_index].append((start, end))
        control_target -= 1

    priority_order = {"high": 0, "medium": 1, "normal": 2}
    candidates.sort(
        key=lambda row: (
            priority_order[row["priority"]],
            row["episode_index"],
            row["start_frame"],
        )
    )
    if maximum_clips is not None:
        candidates = candidates[: max(0, int(maximum_clips))]

    clip_rows = []
    api_rows = []
    for candidate in candidates:
        episode_index = int(candidate["episode_index"])
        route = route_rows[episode_index]
        video_path = (
            dataset
            / "videos"
            / video_key
            / f"chunk-{int(route[f'videos/{video_key}/chunk_index']):03d}"
            / f"file-{int(route[f'videos/{video_key}/file_index']):03d}.mp4"
        ).resolve()
        episode_offset_s = float(route[f"videos/{video_key}/from_timestamp"])
        start_s = candidate["start_frame"] / fps
        end_s = candidate["end_frame"] / fps
        source_start_s = episode_offset_s + start_s
        source_end_s = episode_offset_s + end_s
        clip_id = _clip_id(
            f"{source_dataset or dataset.name}:{supplier_id or 'unknown-supplier'}",
            episode_index,
            candidate["start_frame"],
            candidate["end_frame"],
            candidate["selection_source"],
        )
        clip = {
            "schema_version": SCHEMA_VERSION,
            "clip_id": clip_id,
            "video_id": clip_id,
            "parent_episode_index": episode_index,
            "tasks": route.get("tasks") or [],
            "source_uri": str(video_path),
            "source_dataset": source_dataset or dataset.name,
            "supplier_id": supplier_id,
            # This describes the underlying aggregated MP4 timeline, not the
            # episode-local duration.  It prevents a valid later episode clip
            # from being rejected as out-of-range by downstream readers.
            "duration_s": max(
                float(route[f"videos/{video_key}/to_timestamp"]),
                source_end_s,
            ),
            "fps": fps,
            "clip_start_s": source_start_s,
            "clip_end_s": source_end_s,
            "episode_local_start_s": start_s,
            "episode_local_end_s": end_s,
            **candidate,
            "vla_pretraining": {
                "candidate": True,
                "training_ready": False,
                "split": "unassigned_until_identity_metadata",
                "split_group": clip_id,
                "allowed_objectives": ["video_representation"],
                "loss_masks": {
                    "video_representation": 1,
                    "temporal_prediction": 1,
                    "video_text_alignment": 0,
                    "hand_presence_auxiliary": 0,
                    "mano_motion": 0,
                    "robot_action": 0,
                    "camera_pose": 0,
                    "tactile": 0,
                },
                "clip_sampler": {
                    "mode": "fixed_candidate_window",
                    "fixed_start_s": source_start_s,
                    "window_s": source_end_s - source_start_s,
                    "decode_fps": 8.0,
                },
            },
            "provenance": {
                "quality_root": str(quality_root),
                "code_version": code_version(),
                "raw_immutable": True,
            },
        }
        clip_rows.append(clip)
        # Numeric/schema-only failures do not need a visual teacher.  Keep
        # their clip as review evidence, but do not spend API tokens on it.
        trigger_tasks = list(candidate["candidate_tasks"])
        if not trigger_tasks and candidate["selection_source"] != "random_control_unlabeled":
            continue
        # Once a clip reaches the visual teacher, ask for a broad assessment.
        # Trigger tasks explain why it was recalled; they must not constrain
        # what the model is allowed to discover.
        teacher_tasks = model_tasks
        api_rows.append({
            "request_id": clip_id,
            "schema_version": "egoqc-visual-teacher-request-v1",
            "prompt_version": "egoqc-visual-teacher-v3-open-world",
            "source_uri": str(video_path),
            "clip_start_s": source_start_s,
            "clip_end_s": source_end_s,
            "candidate_tasks": teacher_tasks,
            "trigger_tasks": trigger_tasks,
            "event_codes": candidate["event_codes"],
            "selection_source": candidate["selection_source"],
            "assessment_dimensions": assessment_dimensions,
            "output_path": str(output / "teacher-labels" / clip_id / "teacher-label.json"),
            "required_response": {
                "schema_version": "egoqc-visual-teacher-v1",
                "overall": {
                    "training_usable": "boolean",
                    "recommended_route": "accept|human_review|reject",
                    "confidence": "float[0,1]",
                    "allowed_uses": "list[string]",
                },
                "tasks": {
                    task: {"probability": "float[0,1]", "confidence": "float[0,1]"}
                    for task in teacher_tasks
                },
                "findings": [{
                    "category": "open vocabulary string",
                    "severity": "info|warning|error",
                    "start_s": "number relative to clip",
                    "end_s": "number relative to clip",
                    "evidence": "short factual description",
                    "suggested_action": "accept|review|repair|reject",
                }],
                "missing_annotations": "list[string]",
                "summary": "short factual description",
            },
        })

    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "clip-candidates.jsonl", clip_rows)
    write_jsonl(output / "teacher-api-queue.jsonl", api_rows)
    task_counts = defaultdict(int)
    for row in clip_rows:
        for task in row["candidate_tasks"]:
            task_counts[task] += 1
    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset": str(dataset),
        "quality_root": str(quality_root),
        "fps": fps,
        "bad_frame_events": sum(len(values) for values in events_by_episode.values()),
        "positive_candidates_before_limit": positive_count,
        "clips": len(clip_rows),
        "teacher_api_requests": len(api_rows),
        "requested_random_controls": requested_controls,
        "produced_random_controls": requested_controls - control_target,
        "selection_counts": dict(
            sorted(
                (key, sum(row["selection_source"] == key for row in clip_rows))
                for key in {row["selection_source"] for row in clip_rows}
            )
        ),
        "task_candidate_counts": dict(sorted(task_counts.items())),
        "broad_assessment_dimensions": list(assessment_dimensions),
        "minimum_clip_s": minimum_s,
        "maximum_clip_s": maximum_s,
        "raw_immutable": True,
        "clip_candidates": str(output / "clip-candidates.jsonl"),
        "teacher_api_queue": str(output / "teacher-api-queue.jsonl"),
        "api_credentials_stored": False,
    }
    write_json(output / "summary.json", summary)
    return summary
