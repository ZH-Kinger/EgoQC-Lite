from __future__ import annotations

import hashlib
import itertools
import json
import os
from collections import Counter
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from .canonical import CapabilityManifest, plan_use_cases, route_capabilities
from .provenance import code_version
from .report import write_json
from .video import probe_video


SCHEMA_VERSION = "egoqc-generic-ego-views-v1"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
PARQUET_SCHEMA = pa.schema([
    ("record_id", pa.string()),
    ("video_id", pa.string()),
    ("source_class", pa.string()),
    ("source_dataset", pa.string()),
    ("source_uri", pa.string()),
    ("duration_s", pa.float64()),
    ("fps", pa.float64()),
    ("width", pa.int64()),
    ("height", pa.int64()),
    ("codec", pa.string()),
    ("supplier_id", pa.string()),
    ("person_id", pa.string()),
    ("collection_session_id", pa.string()),
    ("scene_id", pa.string()),
    ("camera_id", pa.string()),
    ("task_id", pa.string()),
    ("training_ready", pa.bool_()),
    ("split", pa.string()),
    ("split_group", pa.string()),
    ("capabilities_json", pa.string()),
    ("capability_route_json", pa.string()),
    ("use_case_eligibility_json", pa.string()),
    ("allowed_objectives_json", pa.string()),
    ("blocked_objectives_json", pa.string()),
    ("issues_json", pa.string()),
    ("source_metadata_json", pa.string()),
    ("source_revision", pa.string()),
    ("code_version", pa.string()),
])


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _ensure_readonly_source_boundary(source: Path, output: Path) -> None:
    if _is_within(output, source):
        raise ValueError(f"输出目录不能位于只读源数据集内: {output}")
    raw_mount = Path("/mnt/data")
    if _is_within(source, raw_mount) and _is_within(output, raw_mount):
        raise ValueError("/mnt/data 原始数据只读；派生产物必须写到 /mnt/workspace 等目录")


def _stable_split(group_id: str) -> str:
    bucket = int(hashlib.sha256(("generic-ego:" + group_id).encode()).hexdigest()[:8], 16) % 1000
    if bucket < 900:
        return "train"
    if bucket < 950:
        return "validation"
    return "test"


def _sidecar(path: Path) -> tuple[Optional[Path], Dict[str, Any], Optional[str]]:
    candidates = (path.with_suffix(".json"), path.with_name(path.name + ".json"))
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            value = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            return candidate, {}, f"{type(error).__name__}: {error}"
        if not isinstance(value, dict):
            return candidate, {}, "sidecar root must be a JSON object"
        return candidate, value, None
    return None, {}, None


def _nonempty(value: Any) -> bool:
    return value not in (None, "", [], {})


def _capabilities(sidecar: Dict[str, Any], probe: Dict[str, Any]) -> CapabilityManifest:
    declared = sidecar.get("capabilities") if isinstance(sidecar.get("capabilities"), dict) else {}
    inferred = {
        "video": bool(probe),
        "video_timestamps": bool(probe.get("time_base")),
        "audio": int(probe.get("audio_streams") or 0) > 0,
        "coarse_activity_labels": any(
            _nonempty(sidecar.get(name)) for name in ("activity", "activities", "category")
        ),
        "task_labels": any(
            _nonempty(sidecar.get(name)) for name in ("task", "task_label", "description")
        ),
        "subtask_labels": any(
            _nonempty(sidecar.get(name)) for name in ("subtasks", "subtask_labels", "clips")
        ),
        "camera_intrinsics": any(
            _nonempty(sidecar.get(name)) for name in ("intrinsics", "camera_intrinsics", "K")
        ),
        "camera_distortion": any(
            _nonempty(sidecar.get(name)) for name in ("distortion", "distortion_coefficients")
        ),
        "camera_trajectory": any(
            _nonempty(sidecar.get(name)) for name in ("camera_trajectory", "camera_poses", "extrinsics")
        ),
        "hand_2d_keypoints": any(
            _nonempty(sidecar.get(name)) for name in ("hand_2d_keypoints", "keypoints_2d")
        ),
        "hand_bounding_boxes": any(
            _nonempty(sidecar.get(name)) for name in ("hand_boxes", "hand_bounding_boxes")
        ),
        "hand_joint_transforms": any(
            _nonempty(sidecar.get(name)) for name in ("hand_joint_transforms", "hand_joints_3d")
        ),
        "mano_parameters": any(
            _nonempty(sidecar.get(name)) for name in ("mano", "mano_parameters", "hand_pose")
        ),
        "prediction_confidence": any(
            _nonempty(sidecar.get(name)) for name in ("confidence", "confidences", "scores")
        ),
        "tactile": _nonempty(sidecar.get("tactile")),
        "depth": any(_nonempty(sidecar.get(name)) for name in ("depth", "depth_path")),
        "imu": any(_nonempty(sidecar.get(name)) for name in ("imu", "imu_path")),
        "multiple_cameras": any(
            _nonempty(sidecar.get(name))
            for name in ("camera_views", "secondary_cameras", "multi_camera_videos")
        ),
        "stereo": bool(sidecar.get("stereo", False)) or _nonempty(sidecar.get("stereo_pair")),
        "camera_extrinsics": any(
            _nonempty(sidecar.get(name))
            for name in ("camera_extrinsics", "rig_extrinsics", "multi_camera_extrinsics")
        ),
        "gaze": any(_nonempty(sidecar.get(name)) for name in ("gaze", "gaze_path")),
        "robot_state": any(
            _nonempty(sidecar.get(name)) for name in ("robot_state", "robot_state_path")
        ),
        "robot_action": any(
            _nonempty(sidecar.get(name)) for name in ("robot_action", "actions", "action_path")
        ),
        "glove_pose": any(
            _nonempty(sidecar.get(name)) for name in ("glove_pose", "glove_path")
        ),
        "privacy_annotations": any(
            _nonempty(sidecar.get(name))
            for name in ("privacy_annotations", "privacy_review", "redaction_manifest")
        ),
        "object_annotations": any(
            _nonempty(sidecar.get(name))
            for name in ("objects", "object_annotations", "object_tracks")
        ),
        "independent_timestamps": any(
            _nonempty(sidecar.get(name)) for name in ("sensor_timestamps", "timestamps")
        ),
        "hand_ground_truth": bool(sidecar.get("hand_ground_truth", False)),
        "trajectory_ground_truth": bool(sidecar.get("trajectory_ground_truth", False)),
    }
    inferred.update({name: bool(value) for name, value in declared.items()})
    return CapabilityManifest.from_dict(inferred)


def inspect_generic_ego_video(
    video: Path,
    *,
    mode: str = "header",
    video_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    video = video.expanduser().resolve()
    if not video.is_file() or video.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"不是支持的 ego 视频: {video}")
    probed, issues = probe_video(video, mode, video_options)
    sidecar_path, metadata, sidecar_error = _sidecar(video)
    capabilities = _capabilities(metadata, probed)
    route = route_capabilities(
        capabilities,
        has_mano_overlay=bool(metadata.get("mano_overlay") or metadata.get("overlay_path")),
    )
    if sidecar_error:
        issues.append({
            "code": "generic_ego_sidecar_invalid",
            "severity": "warning",
            "message": sidecar_error,
            "file": str(sidecar_path),
        })
    issue_rows = [issue.to_dict() if hasattr(issue, "to_dict") else issue for issue in issues]
    return {
        "schema_version": "egoqc-generic-ego-inspection-v1",
        "dataset": str(video),
        "detected_adapter": "generic_ego_raw",
        "compatible": bool(probed),
        "source_readonly": True,
        "video_probe": probed,
        "sidecar_path": str(sidecar_path) if sidecar_path else None,
        "sidecar_metadata": metadata,
        "capabilities": capabilities.to_dict(),
        "capability_route": route,
        "use_case_eligibility": plan_use_cases(capabilities),
        "issues": issue_rows,
    }


def _iter_videos(root: Path, maximum_depth: Optional[int]) -> Iterable[Path]:
    root_depth = len(root.parts)
    for current, directories, filenames in os.walk(root):
        current_path = Path(current)
        depth = len(current_path.parts) - root_depth
        directories[:] = sorted(
            name for name in directories
            if not name.startswith(".") and (maximum_depth is None or depth < maximum_depth)
        )
        for filename in sorted(filenames):
            path = current_path / filename
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                yield path


def _training_contract(
    capabilities: CapabilityManifest,
    group_id: str,
    *,
    technically_usable: bool,
    license_id: Optional[str],
) -> Dict[str, Any]:
    route = route_capabilities(capabilities)
    # RGB and text have a canonical loader today. Arbitrary sidecar MANO/camera/
    # tactile arrays are capabilities, but cannot become targets until a schema-
    # specific normalizer verifies shapes, units, coordinates and timestamps.
    loader_supported = {"video_representation", "temporal_prediction", "video_text_alignment"}
    objectives = [
        name for name, enabled in route["training_objectives"].items()
        if enabled and name in loader_supported
    ]
    masks = {
        name: int(enabled and technically_usable and name in loader_supported)
        for name, enabled in route["training_objectives"].items()
    }
    masks.update({"robot_action": 0, "qc_visual_semantics": 0})
    return {
        "candidate": technically_usable,
        "training_ready": technically_usable and bool(license_id),
        "split": _stable_split(group_id),
        "split_group": group_id,
        "split_warning": None if group_id else "identity_metadata_missing",
        "allowed_objectives": objectives if technically_usable else [],
        "blocked_objectives": {
            name: "requires_schema_specific_canonical_target_adapter"
            for name, enabled in route["training_objectives"].items()
            if enabled and name not in loader_supported
        },
        "loss_masks": masks,
        "target_availability": {
            **route["training_objectives"],
            "robot_action": False,
            "qc_visual_semantics": False,
        },
        "clip_sampler": {
            "mode": "random_window",
            "window_s": 8.0,
            "minimum_visible_duration_s": 0.0,
            "decode_fps": 8.0,
        },
    }


def _record(
    path: Path,
    root: Path,
    source_dataset: str,
    source_class: str,
    license_id: Optional[str],
    mode: str,
    video_options: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    inspection = inspect_generic_ego_video(path, mode=mode, video_options=video_options)
    probe = inspection["video_probe"]
    sidecar = inspection["sidecar_metadata"]
    capabilities = CapabilityManifest.from_dict(inspection["capabilities"])
    relative = path.relative_to(root)
    video_id = relative.with_suffix("").as_posix()
    identity = next(
        (
            f"{field}:{sidecar[field]}"
            for field in ("person_id", "operator_id", "collection_session_id", "session_id")
            if _nonempty(sidecar.get(field))
        ),
        f"video:{video_id}",
    )
    hard_failure = any(issue.get("severity") == "error" for issue in inspection["issues"])
    duration = probe.get("duration")
    if duration is None and probe.get("reported_frames") and probe.get("average_rate"):
        duration = int(probe["reported_frames"]) / float(probe["average_rate"])
    stat = path.stat()
    return {
        "record_id": f"{source_dataset}:{video_id}",
        "video_id": video_id,
        "source_class": source_class,
        "source_dataset": source_dataset,
        "source_uri": str(path.resolve()),
        "duration_s": float(duration or 0.0),
        "fps": probe.get("average_rate"),
        "width": probe.get("width"),
        "height": probe.get("height"),
        "codec": probe.get("codec"),
        "container_format": probe.get("container_format"),
        "task": sidecar.get("task") or sidecar.get("task_label") or sidecar.get("description"),
        "activities": sidecar.get("activities") or sidecar.get("activity"),
        "supplier_id": sidecar.get("supplier_id"),
        "person_id": sidecar.get("person_id"),
        "operator_id": sidecar.get("operator_id"),
        "collection_session_id": sidecar.get("collection_session_id") or sidecar.get("session_id"),
        "scene_id": sidecar.get("scene_id"),
        "camera_id": sidecar.get("camera_id"),
        "task_id": sidecar.get("task_id"),
        "source_metadata": sidecar,
        "capabilities": capabilities.to_dict(),
        "capability_route": inspection["capability_route"],
        "use_case_eligibility": plan_use_cases(capabilities),
        "issues": inspection["issues"],
        "vla_pretraining": _training_contract(
            capabilities,
            identity,
            technically_usable=bool(probe) and not hard_failure,
            license_id=license_id,
        ),
        "provenance": {
            "adapter": "generic_ego_raw",
            "raw_immutable": True,
            "relative_path": relative.as_posix(),
            "source_revision": f"stat:{stat.st_size}:{stat.st_mtime_ns}",
            "sidecar_path": inspection["sidecar_path"],
            "license_id": license_id,
            "code_version": code_version(),
        },
    }


def _parquet_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "record_id": row["record_id"],
            "video_id": row["video_id"],
            "source_class": row["source_class"],
            "source_dataset": row["source_dataset"],
            "source_uri": row["source_uri"],
            "duration_s": row["duration_s"],
            "fps": row["fps"],
            "width": row["width"],
            "height": row["height"],
            "codec": row["codec"],
            "supplier_id": row.get("supplier_id"),
            "person_id": row.get("person_id"),
            "collection_session_id": row.get("collection_session_id"),
            "scene_id": row.get("scene_id"),
            "camera_id": row.get("camera_id"),
            "task_id": row.get("task_id"),
            "training_ready": row["vla_pretraining"]["training_ready"],
            "split": row["vla_pretraining"]["split"],
            "split_group": row["vla_pretraining"]["split_group"],
            "capabilities_json": json.dumps(row["capabilities"], ensure_ascii=False),
            "capability_route_json": json.dumps(row["capability_route"], ensure_ascii=False),
            "use_case_eligibility_json": json.dumps(
                row["use_case_eligibility"], ensure_ascii=False
            ),
            "allowed_objectives_json": json.dumps(
                row["vla_pretraining"]["allowed_objectives"], ensure_ascii=False
            ),
            "blocked_objectives_json": json.dumps(
                row["vla_pretraining"]["blocked_objectives"], ensure_ascii=False
            ),
            "issues_json": json.dumps(row["issues"], ensure_ascii=False),
            "source_metadata_json": json.dumps(row["source_metadata"], ensure_ascii=False),
            "source_revision": row["provenance"]["source_revision"],
            "code_version": row["provenance"]["code_version"],
        }
        for row in rows
    ]


def _bounded_ordered_futures(
    paths: Iterable[Path],
    executor: ThreadPoolExecutor,
    submit,
    maximum_pending: int,
) -> Iterable[tuple[Path, Future]]:
    """Keep discovery deterministic without submitting millions of futures."""

    iterator = iter(paths)
    pending = deque()
    for path in itertools.islice(iterator, maximum_pending):
        pending.append((path, executor.submit(submit, path)))
    while pending:
        path, future = pending.popleft()
        yield path, future
        try:
            next_path = next(iterator)
        except StopIteration:
            continue
        pending.append((next_path, executor.submit(submit, next_path)))


def build_generic_ego_views(
    source_root: Path,
    output: Path,
    *,
    source_dataset: Optional[str] = None,
    source_class: str = "generic_ego",
    license_id: Optional[str] = None,
    workers: int = 16,
    maximum_depth: Optional[int] = None,
    limit: Optional[int] = None,
    video_check: str = "header",
    video_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    source_root = source_root.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source_root.is_dir():
        raise NotADirectoryError(source_root)
    if workers < 1 or (limit is not None and limit < 1):
        raise ValueError("workers 和 limit 必须大于 0")
    if maximum_depth is not None and maximum_depth < 0:
        raise ValueError("maximum_depth 必须 >= 0")
    _ensure_readonly_source_boundary(source_root, output)
    dataset_name = source_dataset or source_root.name
    output.mkdir(parents=True, exist_ok=True)
    jsonl_path = output / "generic-ego.jsonl"
    parquet_path = output / "generic-ego.parquet"
    errors_path = output / "errors.jsonl"
    jsonl_temp = jsonl_path.with_name(f".{jsonl_path.name}.{os.getpid()}.tmp")
    parquet_temp = parquet_path.with_name(f".{parquet_path.name}.{os.getpid()}.tmp")
    errors_temp = errors_path.with_name(f".{errors_path.name}.{os.getpid()}.tmp")
    paths: Iterable[Path] = _iter_videos(source_root, maximum_depth)
    if limit is not None:
        paths = itertools.islice(paths, limit)
    records = 0
    errors = 0
    discovered = 0
    training_ready = 0
    technical_candidates = 0
    capability_counts: Counter = Counter()
    objective_counts: Counter = Counter()
    use_case_counts: Counter = Counter()
    parquet_buffer: List[Dict[str, Any]] = []
    parquet_writer = pq.ParquetWriter(parquet_temp, PARQUET_SCHEMA, compression="zstd")

    def submit(path: Path) -> Dict[str, Any]:
        return _record(
            path,
            source_root,
            dataset_name,
            source_class,
            license_id,
            video_check,
            video_options,
        )

    try:
        with jsonl_temp.open("w", encoding="utf-8") as json_handle, errors_temp.open(
            "w", encoding="utf-8"
        ) as error_handle, ThreadPoolExecutor(max_workers=workers) as executor:
            for path, future in _bounded_ordered_futures(
                paths, executor, submit, maximum_pending=max(workers * 4, 16)
            ):
                discovered += 1
                try:
                    row = future.result()
                except Exception as error:
                    errors += 1
                    error_handle.write(json.dumps({
                        "source_uri": str(path),
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "raw_immutable": True,
                    }, ensure_ascii=False) + "\n")
                    continue
                records += 1
                training_ready += int(row["vla_pretraining"]["training_ready"])
                technical_candidates += int(row["vla_pretraining"]["candidate"])
                capability_counts.update(
                    name for name, value in row["capabilities"].items() if value
                )
                objective_counts.update(row["vla_pretraining"]["allowed_objectives"])
                use_case_counts.update(
                    f"{name}:{value['status']}"
                    for name, value in row["use_case_eligibility"].items()
                )
                json_handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                parquet_buffer.extend(_parquet_rows([row]))
                if len(parquet_buffer) >= 4096:
                    parquet_writer.write_table(pa.Table.from_pylist(parquet_buffer, schema=PARQUET_SCHEMA))
                    parquet_buffer.clear()
        if parquet_buffer:
            parquet_writer.write_table(pa.Table.from_pylist(parquet_buffer, schema=PARQUET_SCHEMA))
            parquet_buffer.clear()
    finally:
        parquet_writer.close()
    jsonl_temp.replace(jsonl_path)
    parquet_temp.replace(parquet_path)
    errors_temp.replace(errors_path)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "source_root": str(source_root),
        "source_dataset": dataset_name,
        "source_class": source_class,
        "source_readonly": True,
        "videos_discovered": discovered,
        "records": records,
        "errors": errors,
        "training_ready": training_ready,
        "technical_candidates": technical_candidates,
        "streaming": {
            "bounded_pending_futures": max(workers * 4, 16),
            "parquet_row_group_buffer": 4096,
            "memory_scales_with_total_file_count": False
        },
        "capability_counts": dict(capability_counts),
        "training_objective_counts": dict(objective_counts),
        "use_case_status_counts": dict(use_case_counts),
        "missing_optional_modalities_are_failures": False,
        "artifacts": {
            "jsonl": str(output / "generic-ego.jsonl"),
            "parquet": str(output / "generic-ego.parquet"),
            "errors": str(output / "errors.jsonl"),
        },
    }
    write_json(output / "summary.json", summary)
    return summary
