from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .math3d import (
    euler_xyz_to_matrix,
    geodesic_degrees,
    longest_false_run,
    so3_errors,
    transform_points,
)
from .sampling import select_sample_frames
from .temporal import analyze_temporal_quality
from .types import EpisodeResult, Issue


def _arrays(table: pa.Table, name: str, dtype=np.float64) -> np.ndarray:
    array = table[name].combine_chunks()
    if pa.types.is_fixed_size_list(array.type):
        values = array.values.to_numpy(zero_copy_only=False)
        return np.asarray(values, dtype=dtype).reshape(len(array), array.type.list_size)
    if pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        offsets = np.asarray(array.offsets.to_numpy(zero_copy_only=False), dtype=np.int64)
        widths = np.diff(offsets)
        if len(widths) == 0:
            return np.empty((0, 0), dtype=dtype)
        if np.all(widths == widths[0]):
            values = array.values.to_numpy(zero_copy_only=False)
            return np.asarray(values, dtype=dtype).reshape(len(array), int(widths[0]))
        return np.asarray(array.to_pylist(), dtype=dtype)
    return np.asarray(array.to_numpy(zero_copy_only=False), dtype=dtype)


def _percentile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, q)) if values.size else float("nan")


def _severity(value: float, warning: float, error: float) -> str:
    if not np.isfinite(value):
        return "error"
    if value > error:
        return "error"
    if value > warning:
        return "warning"
    return "info"


def _true_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return half-open [start, end) runs without per-frame Python iteration."""
    values = np.asarray(mask, dtype=bool).reshape(-1)
    if values.size == 0:
        return []
    padded = np.concatenate(([False], values, [False])).astype(np.int8)
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def analyze_hand_visibility(
    state_mask: np.ndarray,
    fps: float,
    minimum_continuous_s: float = 5.0,
) -> Dict[str, Any]:
    """Measure task-level any-hand visibility using the acceptance definition.

    Leading/trailing absence is reported separately. Only gaps bounded by visible
    hand frames count as "hand left the view", which avoids rejecting normal trim
    margins. Effective duration starts after the first ``minimum_continuous_s``
    seconds of every qualifying visible run.
    """
    masks = np.asarray(state_mask, dtype=bool)
    if masks.ndim != 2 or masks.shape[1] != 2:
        raise ValueError("state_mask 必须是 [frames, 2]")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps 必须为正数")
    minimum_continuous_s = max(0.0, float(minimum_continuous_s))
    visible = np.any(masks, axis=1)
    count = int(visible.size)
    visible_runs = _true_runs(visible)
    missing_runs = _true_runs(~visible)
    first = visible_runs[0][0] if visible_runs else None
    last_exclusive = visible_runs[-1][1] if visible_runs else None
    internal_missing = [
        (start, end)
        for start, end in missing_runs
        if first is not None and start > first and end < last_exclusive
    ]
    longest_internal_frames = max((end - start for start, end in internal_missing), default=0)
    visible_lengths = [end - start for start, end in visible_runs]
    minimum_frames = minimum_continuous_s * fps
    qualifying = [length for length in visible_lengths if length >= minimum_frames]
    return {
        "any_hand_valid_ratio": float(np.mean(visible)) if count else 0.0,
        "first_hand_valid_frame": first,
        "last_hand_valid_frame": (last_exclusive - 1) if last_exclusive is not None else None,
        "leading_hand_missing_s": float(first / fps) if first is not None else float(count / fps),
        "trailing_hand_missing_s": (
            float((count - last_exclusive) / fps) if last_exclusive is not None else float(count / fps)
        ),
        "internal_hand_missing_gap_count": len(internal_missing),
        "longest_internal_hand_missing_gap_frames": longest_internal_frames,
        "longest_internal_hand_missing_gap_s": float(longest_internal_frames / fps),
        "continuous_hand_visible_segment_count": len(visible_runs),
        "qualifying_hand_visible_segment_count": len(qualifying),
        "longest_continuous_hand_visible_s": float(max(visible_lengths, default=0) / fps),
        "qualified_visible_duration_s": float(sum(qualifying) / fps),
        "effective_video_duration_s": float(
            sum(max(0.0, length / fps - minimum_continuous_s) for length in qualifying)
        ),
    }


def _validate_frame_schema(table: pa.Table, config: Dict[str, Any]) -> List[Issue]:
    issues: List[Issue] = []
    expected_types = {"float64": pa.float64(), "int64": pa.int64(), "bool": pa.bool_()}
    for name, expected in config.get("frame_schema", {}).items():
        if name not in table.column_names:
            continue
        column_type = table.schema.field(name).type
        is_list = (
            pa.types.is_list(column_type)
            or pa.types.is_large_list(column_type)
            or pa.types.is_fixed_size_list(column_type)
        )
        value_type = column_type.value_type if is_list else column_type
        expected_type = expected_types.get(str(expected.get("dtype")))
        if expected_type is not None and value_type != expected_type:
            issues.append(
                Issue(
                    "frame_dtype_mismatch",
                    "error",
                    f"{name} parquet dtype={column_type}，期望 {expected.get('dtype')}",
                )
            )
        expected_width = int(list(expected.get("shape", [1]))[0])
        if expected_width == 1:
            actual_width = column_type.list_size if pa.types.is_fixed_size_list(column_type) else (None if is_list else 1)
        elif pa.types.is_fixed_size_list(column_type):
            actual_width = column_type.list_size
        elif is_list:
            widths = {len(value) for value in table[name].to_pylist() if value is not None}
            actual_width = next(iter(widths)) if len(widths) == 1 else None
        else:
            actual_width = 1
        if actual_width != expected_width:
            issues.append(
                Issue(
                    "frame_shape_mismatch",
                    "error",
                    f"{name} parquet width={actual_width}，期望 {expected_width}",
                )
            )
    return issues


def load_episode_index(dataset: Path) -> pa.Table:
    paths = sorted((dataset / "meta" / "episodes").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError("meta/episodes 下没有 parquet")
    return pa.concat_tables([pq.read_table(path) for path in paths])


def load_task_map(dataset: Path) -> Dict[int, str]:
    table = pq.read_table(dataset / "meta" / "tasks.parquet")
    text_columns = [name for name in ("task", "tasks", "text") if name in table.column_names]
    if "task_index" not in table.column_names or not text_columns:
        return {}
    return {
        int(index): str(text)
        for index, text in zip(
            table["task_index"].to_pylist(),
            table[text_columns[0]].to_pylist(),
        )
    }


def validate_dataset_structure(dataset: Path, config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Issue]]:
    issues: List[Issue] = []
    info_path = dataset / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(info_path)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    for required in ("meta", "data", "videos"):
        if not (dataset / required).exists():
            issues.append(Issue("missing_directory", "error", f"缺少目录: {required}", file=required))

    features = info.get("features", {})
    schema = config.get("feature_schema", {})
    for name, expected in schema.items():
        declared = features.get(name)
        if declared is None:
            issues.append(Issue("feature_not_declared", "error", f"info.json 未声明 feature: {name}"))
            continue
        actual_dtype = str(declared.get("dtype", ""))
        if actual_dtype != str(expected.get("dtype", "")):
            issues.append(
                Issue(
                    "feature_dtype_mismatch",
                    "error",
                    f"info.json {name} dtype={actual_dtype}，期望 {expected.get('dtype')}",
                )
            )
        actual_shape = list(declared.get("shape", []))
        if actual_shape != list(expected.get("shape", [])):
            issues.append(
                Issue(
                    "feature_shape_mismatch",
                    "error",
                    f"info.json {name} shape={actual_shape}，期望 {expected.get('shape')}",
                )
            )
    video_key = config.get("video_key", "observation.images.ego")
    video_feature = features.get(video_key)
    if video_feature is None:
        issues.append(Issue("feature_not_declared", "error", f"info.json 未声明 video feature: {video_key}"))
    else:
        if video_feature.get("dtype") != "video":
            issues.append(Issue("feature_dtype_mismatch", "error", f"{video_key} dtype 必须为 video"))
        shape = list(video_feature.get("shape", []))
        if len(shape) != 3 or shape[-1] != 3:
            issues.append(Issue("feature_shape_mismatch", "error", f"{video_key} shape 必须为 [height,width,3]"))
    for name in config["required_frame_columns"]:
        if name not in features and name not in {
            "left_transl_world", "left_orient_world", "left_hand_pose", "left_kept",
            "right_transl_world", "right_orient_world", "right_hand_pose", "right_kept",
            "left_seg_start", "left_seg_end", "right_seg_start", "right_seg_end",
        }:
            issues.append(
                Issue("feature_not_declared", "warning", f"info.json 未声明 feature: {name}")
            )
    tasks_path = dataset / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        issues.append(Issue("missing_metadata_file", "error", "缺少 meta/tasks.parquet", file=str(tasks_path)))
    else:
        try:
            task_schema = set(pq.ParquetFile(tasks_path).schema_arrow.names)
            if "task_index" not in task_schema:
                issues.append(Issue("missing_metadata_column", "error", "tasks.parquet 缺少 task_index", file=str(tasks_path)))
            if not ({"task", "tasks", "text"} & task_schema):
                issues.append(Issue("missing_metadata_column", "error", "tasks.parquet 缺少任务文本列", file=str(tasks_path)))
        except Exception as error:
            issues.append(Issue("metadata_read_failed", "error", f"tasks.parquet 无法读取: {error}", file=str(tasks_path)))
    episode_paths = sorted((dataset / "meta" / "episodes").rglob("*.parquet"))
    if episode_paths:
        try:
            episode_schema = set(pq.ParquetFile(episode_paths[0]).schema_arrow.names)
            for column in config.get("required_episode_columns", []):
                if column not in episode_schema:
                    issues.append(
                        Issue(
                            "missing_metadata_column",
                            "error",
                            f"episode 索引缺少 {column}",
                            file=str(episode_paths[0]),
                        )
                    )
        except Exception as error:
            issues.append(Issue("metadata_read_failed", "error", f"episode 索引无法读取: {error}", file=str(episode_paths[0])))
    return info, issues


def validate_episode(
    table: pa.Table,
    episode_index: int,
    expected_length: int,
    fps: float,
    config: Dict[str, Any],
    file_name: str,
    filtered: bool = False,
    expected_from_index: Optional[int] = None,
    expected_to_index: Optional[int] = None,
    task_map: Optional[Dict[int, str]] = None,
    expected_tasks: Optional[List[str]] = None,
) -> EpisodeResult:
    result = EpisodeResult(episode_index=episode_index, length=expected_length)
    columns = set(table.column_names)
    missing = sorted(set(config["required_frame_columns"]) - columns)
    if missing:
        result.issues.append(
            Issue(
                "missing_columns",
                "error",
                f"逐帧 parquet 缺少字段: {', '.join(missing)}",
                episode_index,
                file_name,
            )
        )
        result.tier = "quarantine"
        return result

    schema_issues = _validate_frame_schema(table, config)
    if schema_issues:
        for issue in schema_issues:
            issue.episode_index = episode_index
            issue.file = file_name
        result.issues.extend(schema_issues)
        result.tier = "quarantine"
        return result

    if filtered:
        ep = table
    else:
        ep_values = _arrays(table, "episode_index", np.int64).reshape(-1)
        ep = table.filter(pa.array(ep_values == episode_index))
    n = len(ep)
    result.metrics["actual_length"] = n
    if n != expected_length:
        result.issues.append(
            Issue(
                "episode_length_mismatch",
                "error",
                f"metadata length={expected_length}, parquet rows={n}",
                episode_index,
                file_name,
            )
        )
    if n == 0:
        result.tier = "quarantine"
        return result

    frame_index = _arrays(ep, "frame_index", np.int64).reshape(-1)
    if not np.array_equal(frame_index, np.arange(n)):
        result.issues.append(
            Issue("frame_index_not_contiguous", "error", "frame_index 不是从 0 连续递增", episode_index, file_name)
        )

    global_index = _arrays(ep, "index", np.int64).reshape(-1)
    if len(global_index) > 1 and not np.all(np.diff(global_index) == 1):
        result.issues.append(
            Issue("global_index_not_contiguous", "error", "episode 内全局 index 不是连续递增", episode_index, file_name)
        )
    if expected_from_index is not None and expected_to_index is not None:
        if (
            int(global_index[0]) != int(expected_from_index)
            or int(global_index[-1]) + 1 != int(expected_to_index)
            or int(expected_to_index) - int(expected_from_index) != n
        ):
            result.issues.append(
                Issue(
                    "global_index_range_mismatch",
                    "error",
                    f"index [{int(global_index[0])},{int(global_index[-1]) + 1}) 与 metadata [{expected_from_index},{expected_to_index}) 不一致",
                    episode_index,
                    file_name,
                )
            )

    timestamp = _arrays(ep, "timestamp").reshape(-1)
    expected_time = frame_index / fps
    timestamp_error = float(np.nanmax(np.abs(timestamp - expected_time)))
    result.metrics["timestamp_max_error_s"] = timestamp_error
    if timestamp_error > config["thresholds"]["timestamp_tolerance_s"]:
        result.issues.append(
            Issue(
                "timestamp_mismatch",
                "warning",
                f"timestamp 最大误差 {timestamp_error:.6f}s",
                episode_index,
                file_name,
            )
        )

    if task_map is not None:
        episode_task_indices = set(
            _arrays(ep, "task_index", np.int64).reshape(-1).tolist()
        )
        unknown = sorted(episode_task_indices - set(task_map))
        if unknown:
            result.issues.append(
                Issue("task_index_unknown", "error", f"task_index 不在 tasks.parquet: {unknown}", episode_index, file_name)
            )
        if expected_tasks:
            mapped = {task_map[index] for index in episode_task_indices if index in task_map}
            if mapped != {str(task) for task in expected_tasks}:
                result.issues.append(
                    Issue(
                        "task_text_invalid",
                        "error",
                        f"episode tasks={expected_tasks}，逐帧映射={sorted(mapped)}",
                        episode_index,
                        file_name,
                    )
                )

    state = _arrays(ep, "observation.state")
    state_mask = _arrays(ep, "state_mask", bool)
    intrinsics = _arrays(ep, "intrinsics")
    extr = _arrays(ep, "extrinsics_w2c").reshape(n, 4, 4)
    left_world = _arrays(ep, "left_transl_world").reshape(n, 3)
    right_world = _arrays(ep, "right_transl_world").reshape(n, 3)
    left_orient = _arrays(ep, "left_orient_world").reshape(n, 3, 3)
    right_orient = _arrays(ep, "right_orient_world").reshape(n, 3, 3)
    left_pose = _arrays(ep, "left_hand_pose").reshape(n, 15, 3, 3)
    right_pose = _arrays(ep, "right_hand_pose").reshape(n, 15, 3, 3)
    left_kept = _arrays(ep, "left_kept", bool).reshape(-1)
    right_kept = _arrays(ep, "right_kept", bool).reshape(-1)
    for side in ("left", "right"):
        for suffix in ("seg_start", "seg_end"):
            values = _arrays(ep, f"{side}_{suffix}", np.int64).reshape(-1)
            invalid = int(np.count_nonzero(values != -1))
            if invalid:
                result.issues.append(
                    Issue(
                        "segment_marker_invalid",
                        "error",
                        f"{side}_{suffix} 有 {invalid} 帧不是 -1",
                        episode_index,
                        file_name,
                    )
                )

    kept_mismatch = int(
        np.count_nonzero(left_kept != state_mask[:, 0])
        + np.count_nonzero(right_kept != state_mask[:, 1])
    )
    result.metrics["kept_mask_mismatch_count"] = kept_mismatch
    if kept_mismatch:
        result.issues.append(
            Issue(
                "kept_mask_mismatch",
                "error",
                f"left/right_kept 与 state_mask 不一致，共 {kept_mismatch} 个手帧",
                episode_index,
                file_name,
            )
        )

    visibility_config = config.get("visibility", {})
    maximum_absence_s = float(visibility_config.get("maximum_internal_absence_s", 1.0))
    minimum_continuous_s = float(visibility_config.get("minimum_continuous_visible_s", 5.0))
    visibility = analyze_hand_visibility(state_mask, fps, minimum_continuous_s)
    result.metrics.update(visibility)
    if visibility["longest_internal_hand_missing_gap_s"] > maximum_absence_s:
        result.issues.append(
            Issue(
                "hand_out_of_view_too_long",
                "error",
                "任务中手连续离开画面 "
                f"{visibility['longest_internal_hand_missing_gap_s']:.3f}s，"
                f"超过 {maximum_absence_s:.3f}s",
                episode_index,
                file_name,
                {
                    "measured_s": visibility["longest_internal_hand_missing_gap_s"],
                    "maximum_s": maximum_absence_s,
                    "gap_count": visibility["internal_hand_missing_gap_count"],
                },
            )
        )
    episode_duration_s = n / fps
    if (
        episode_duration_s >= minimum_continuous_s
        and visibility["qualifying_hand_visible_segment_count"] == 0
    ):
        result.issues.append(
            Issue(
                "insufficient_continuous_hand_visibility",
                "error",
                f"没有连续可见达到 {minimum_continuous_s:.3f}s 的手部片段",
                episode_index,
                file_name,
                {
                    "longest_visible_s": visibility["longest_continuous_hand_visible_s"],
                    "minimum_s": minimum_continuous_s,
                },
            )
        )

    if "main_type" in columns:
        main_type = _arrays(ep, "main_type", np.int64).reshape(-1)
        invalid_main_type = int(np.count_nonzero(~np.isin(main_type, (-1, 0, 1))))
        result.metrics["invalid_main_type_count"] = invalid_main_type
        if invalid_main_type:
            result.issues.append(
                Issue(
                    "invalid_main_type",
                    "warning",
                    f"main_type 有 {invalid_main_type} 帧不在 -1/0/1 中",
                    episode_index,
                    file_name,
                )
            )

    finite_ratio = float(
        np.mean(
            np.isfinite(state).all(axis=1)
            & np.isfinite(intrinsics).all(axis=1)
            & np.isfinite(extr).all(axis=(1, 2))
            & np.isfinite(left_world).all(axis=1)
            & np.isfinite(right_world).all(axis=1)
        )
    )
    result.metrics["finite_frame_ratio"] = finite_ratio
    if finite_ratio < 1.0:
        result.issues.append(
            Issue("non_finite_values", "error", f"有限值帧比例 {finite_ratio:.3%}", episode_index, file_name)
        )

    rotations = np.concatenate(
        [
            extr[:, None, :3, :3],
            left_orient[:, None],
            right_orient[:, None],
            left_pose,
            right_pose,
        ],
        axis=1,
    )
    orth, det = so3_errors(rotations)
    orth_p99, det_p99 = _percentile(orth, 99), _percentile(det, 99)
    result.metrics.update(
        {"so3_orthogonality_p99": orth_p99, "so3_determinant_error_p99": det_p99}
    )
    if (
        orth_p99 > config["thresholds"]["so3_orthogonality_error"]
        or det_p99 > config["thresholds"]["so3_determinant_error"]
    ):
        result.issues.append(
            Issue(
                "invalid_rotation_matrix",
                "error",
                f"SO(3) 异常: orth_p99={orth_p99:.5g}, det_error_p99={det_p99:.5g}",
                episode_index,
                file_name,
            )
        )

    camera_rotation = extr[:, :3, :3]
    sides = {
        "left": (0, left_world, left_orient, left_pose, state[:, 0:3], state[:, 3:6], state[:, 6:51], state[:, 51:61]),
        "right": (1, right_world, right_orient, right_pose, state[:, 61:64], state[:, 64:67], state[:, 67:112], state[:, 112:122]),
    }
    for side, (slot, world, orient, pose, stored_pos, stored_euler, stored_pose_euler, betas) in sides.items():
        valid = state_mask[:, slot]
        valid_ratio = float(np.mean(valid))
        result.metrics[f"{side}_valid_ratio"] = valid_ratio
        result.metrics[f"{side}_longest_missing_gap"] = longest_false_run(valid)
        if valid_ratio < config["thresholds"]["minimum_valid_hand_ratio"]:
            result.issues.append(
                Issue("low_valid_ratio", "warning", f"{side} 有效率仅 {valid_ratio:.2%}", episode_index, file_name)
            )
        if not np.any(valid):
            continue

        expected_pos = transform_points(extr, world)
        pos_error = np.linalg.norm(expected_pos - stored_pos, axis=1)[valid]
        pos_p95 = _percentile(pos_error, 95)
        result.metrics[f"{side}_position_error_p95_m"] = pos_p95
        pos_severity = _severity(
            pos_p95,
            config["thresholds"]["position_warning_m"],
            config["thresholds"]["position_error_m"],
        )
        if pos_severity != "info":
            result.issues.append(
                Issue(
                    "world_camera_position_mismatch",
                    pos_severity,
                    f"{side} wrist 世界系→相机系误差 p95={pos_p95:.4f}m",
                    episode_index,
                    file_name,
                )
            )

        expected_rot = camera_rotation @ orient
        stored_rot = euler_xyz_to_matrix(stored_euler)
        rot_error = geodesic_degrees(expected_rot, stored_rot)[valid]
        rot_p95 = _percentile(rot_error, 95)
        result.metrics[f"{side}_rotation_error_p95_deg"] = rot_p95
        rot_severity = _severity(
            rot_p95,
            config["thresholds"]["rotation_warning_deg"],
            config["thresholds"]["rotation_error_deg"],
        )
        if rot_severity != "info":
            result.issues.append(
                Issue(
                    "world_camera_rotation_mismatch",
                    rot_severity,
                    f"{side} wrist 旋转误差 p95={rot_p95:.2f}°",
                    episode_index,
                    file_name,
                )
            )

        stored_pose = euler_xyz_to_matrix(stored_pose_euler.reshape(n, 15, 3))
        pose_error = geodesic_degrees(pose, stored_pose)[valid]
        pose_p95 = _percentile(pose_error, 95)
        result.metrics[f"{side}_pose_repr_error_p95_deg"] = pose_p95
        pose_severity = _severity(
            pose_p95,
            config["thresholds"]["pose_repr_warning_deg"],
            config["thresholds"]["pose_repr_error_deg"],
        )
        if pose_severity != "info":
            result.issues.append(
                Issue(
                    "pose_representation_mismatch",
                    pose_severity,
                    f"{side} rotmat↔Euler 误差 p95={pose_p95:.2f}°",
                    episode_index,
                    file_name,
                )
            )

        beta_std = float(np.nanmax(np.nanstd(betas, axis=0)))
        result.metrics[f"{side}_beta_max_std"] = beta_std
        if beta_std > config["thresholds"]["beta_std_warning"]:
            result.issues.append(
                Issue("beta_drift", "warning", f"{side} MANO betas 最大 std={beta_std:.4f}", episode_index, file_name)
            )

    temporal_metrics, temporal_issues, temporal_frames = analyze_temporal_quality(
        left_world,
        right_world,
        left_orient,
        right_orient,
        left_pose,
        right_pose,
        state_mask,
        extr,
        fps,
        config,
        episode_index,
        file_name,
    )
    result.metrics.update(temporal_metrics)
    result.issues.extend(temporal_issues)
    result.sample_frames = select_sample_frames(
        n,
        config["sampling"],
        left_world,
        right_world,
        state_mask,
        fps,
        episode_index,
        temporal_frames,
    )
    result.tier = classify(result.issues)
    return result


def classify(issues: Iterable[Issue]) -> str:
    severities = {issue.severity for issue in issues}
    codes = {issue.code for issue in issues}
    hard = {
        "missing_columns",
        "episode_length_mismatch",
        "frame_index_not_contiguous",
        "non_finite_values",
        "invalid_rotation_matrix",
        "kept_mask_mismatch",
    }
    if codes & hard:
        return "quarantine"
    if "error" in severities:
        return "bronze"
    if "warning" in severities:
        return "silver"
    return "gold"
