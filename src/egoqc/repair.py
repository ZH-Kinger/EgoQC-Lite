from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .annotated_video import _episode_route, _fps, render_annotated_episode
from .mano import ManoOverlayRenderer
from .math3d import geodesic_degrees, matrix_to_euler_xyz, transform_points
from .temporal import analyze_temporal_quality


def _alpha(cutoff: np.ndarray, dt: float) -> np.ndarray:
    cutoff = np.maximum(np.asarray(cutoff, dtype=np.float64), 1e-9)
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


def _true_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return half-open continuous true runs; invalid gaps stay untouched."""

    mask = np.asarray(mask, dtype=bool)
    changes = np.diff(np.pad(mask.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def one_euro_vectors(
    values: np.ndarray,
    fps: float,
    min_cutoff: float,
    beta: float,
    derivative_cutoff: float,
) -> np.ndarray:
    """One-Euro filtering with one adaptive cutoff per vector."""

    values = np.asarray(values, dtype=np.float64)
    output = values.copy()
    if len(values) < 2:
        return output
    dt = 1.0 / float(fps)
    derivative_hat = np.zeros(values.shape[1:], dtype=np.float64)
    previous_input = values[0].copy()
    previous_output = values[0].copy()
    derivative_alpha = float(_alpha(np.asarray(derivative_cutoff), dt))
    for index in range(1, len(values)):
        derivative = (values[index] - previous_input) / dt
        derivative_hat += derivative_alpha * (derivative - derivative_hat)
        speed = float(np.linalg.norm(derivative_hat))
        value_alpha = float(_alpha(np.asarray(min_cutoff + beta * speed), dt))
        previous_output += value_alpha * (values[index] - previous_output)
        output[index] = previous_output
        previous_input = values[index]
    return output


def _matrix_to_quaternion(matrices: np.ndarray) -> np.ndarray:
    """Convert proper rotation matrices to normalized xyzw quaternions."""

    matrices = np.asarray(matrices, dtype=np.float64)
    r00, r11, r22 = matrices[..., 0, 0], matrices[..., 1, 1], matrices[..., 2, 2]
    x = 0.5 * np.sqrt(np.maximum(0.0, 1.0 + r00 - r11 - r22))
    y = 0.5 * np.sqrt(np.maximum(0.0, 1.0 - r00 + r11 - r22))
    z = 0.5 * np.sqrt(np.maximum(0.0, 1.0 - r00 - r11 + r22))
    w = 0.5 * np.sqrt(np.maximum(0.0, 1.0 + r00 + r11 + r22))
    x = np.copysign(x, matrices[..., 2, 1] - matrices[..., 1, 2])
    y = np.copysign(y, matrices[..., 0, 2] - matrices[..., 2, 0])
    z = np.copysign(z, matrices[..., 1, 0] - matrices[..., 0, 1])
    quaternion = np.stack([x, y, z, w], axis=-1)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    return quaternion / np.maximum(norm, 1e-12)


def _quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion = quaternion / np.maximum(np.linalg.norm(quaternion, axis=-1, keepdims=True), 1e-12)
    x, y, z, w = np.moveaxis(quaternion, -1, 0)
    output = np.empty(quaternion.shape[:-1] + (3, 3), dtype=np.float64)
    output[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    output[..., 0, 1] = 2.0 * (x * y - z * w)
    output[..., 0, 2] = 2.0 * (x * z + y * w)
    output[..., 1, 0] = 2.0 * (x * y + z * w)
    output[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    output[..., 1, 2] = 2.0 * (y * z - x * w)
    output[..., 2, 0] = 2.0 * (x * z - y * w)
    output[..., 2, 1] = 2.0 * (y * z + x * w)
    output[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return output


def _slerp(previous: np.ndarray, target: np.ndarray, amount: np.ndarray) -> np.ndarray:
    dot = np.sum(previous * target, axis=-1, keepdims=True)
    target = np.where(dot < 0.0, -target, target)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    angle = np.arccos(dot)
    sine = np.sin(angle)
    amount = np.asarray(amount, dtype=np.float64)[..., None]
    linear = previous + amount * (target - previous)
    curved = (
        np.sin((1.0 - amount) * angle) / np.maximum(sine, 1e-12) * previous
        + np.sin(amount * angle) / np.maximum(sine, 1e-12) * target
    )
    result = np.where(np.abs(sine) < 1e-6, linear, curved)
    return result / np.maximum(np.linalg.norm(result, axis=-1, keepdims=True), 1e-12)


def one_euro_rotations(
    matrices: np.ndarray,
    fps: float,
    min_cutoff: float,
    beta: float,
    derivative_cutoff: float,
) -> np.ndarray:
    """Adaptive quaternion SLERP over one or more SO(3) streams."""

    matrices = np.asarray(matrices, dtype=np.float64)
    output = matrices.copy()
    if len(matrices) < 2:
        return output
    quaternions = _matrix_to_quaternion(matrices)
    filtered = quaternions.copy()
    dt = 1.0 / float(fps)
    derivative_hat = np.zeros(quaternions.shape[1:-1], dtype=np.float64)
    derivative_alpha = float(_alpha(np.asarray(derivative_cutoff), dt))
    previous_input = quaternions[0].copy()
    previous_output = quaternions[0].copy()
    for index in range(1, len(quaternions)):
        current = quaternions[index].copy()
        current = np.where(
            (np.sum(previous_input * current, axis=-1) < 0.0)[..., None], -current, current
        )
        dot = np.clip(np.abs(np.sum(previous_input * current, axis=-1)), 0.0, 1.0)
        angular_speed = 2.0 * np.arccos(dot) / dt
        derivative_hat += derivative_alpha * (angular_speed - derivative_hat)
        cutoff = min_cutoff + beta * np.abs(derivative_hat)
        previous_output = _slerp(previous_output, current, _alpha(cutoff, dt))
        filtered[index] = previous_output
        previous_input = current
    return _quaternion_to_matrix(filtered)


def repair_episode_records(
    records: Sequence[Dict[str, Any]], fps: float, repair_config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Return repaired copies while preserving masks, gaps, betas and metadata."""

    repaired = [dict(record) for record in records]
    masks = np.asarray([record["state_mask"] for record in records], dtype=bool).reshape(-1, 2)
    for side_index, side in enumerate(("left", "right")):
        if records and f"{side}_kept" in records[0]:
            masks[:, side_index] &= np.asarray(
                [record[f"{side}_kept"] for record in records], dtype=bool
            )
    extrinsics = np.asarray([record["extrinsics_w2c"] for record in records], dtype=np.float64).reshape(-1, 4, 4)
    defaults = {"min_cutoff": 2.0, "beta": 0.7, "derivative_cutoff": 1.0}
    position_options = {**defaults, **repair_config.get("position", {})}
    rotation_options = {**defaults, **repair_config.get("rotation", {})}
    for side_index, side in enumerate(("left", "right")):
        positions = np.asarray([record[f"{side}_transl_world"] for record in records], dtype=np.float64)
        wrists = np.asarray([record[f"{side}_orient_world"] for record in records], dtype=np.float64).reshape(-1, 3, 3)
        poses = np.asarray([record[f"{side}_hand_pose"] for record in records], dtype=np.float64).reshape(-1, 15, 3, 3)
        for start, stop in _true_runs(masks[:, side_index]):
            positions[start:stop] = one_euro_vectors(positions[start:stop], fps, **position_options)
            wrists[start:stop] = one_euro_rotations(wrists[start:stop], fps, **rotation_options)
            poses[start:stop] = one_euro_rotations(poses[start:stop], fps, **rotation_options)
        for frame, record in enumerate(repaired):
            record[f"{side}_transl_world"] = positions[frame].tolist()
            record[f"{side}_orient_world"] = wrists[frame].reshape(-1).tolist()
            record[f"{side}_hand_pose"] = poses[frame].reshape(-1).tolist()
            state = np.asarray(record["observation.state"], dtype=np.float64).copy()
            offset = 0 if side == "left" else 61
            if masks[frame, side_index]:
                state[offset : offset + 3] = transform_points(extrinsics[frame], positions[frame])
                camera_wrist = extrinsics[frame, :3, :3] @ wrists[frame]
                state[offset + 3 : offset + 6] = matrix_to_euler_xyz(camera_wrist)
                state[offset + 6 : offset + 51] = matrix_to_euler_xyz(poses[frame]).reshape(-1)
            record["observation.state"] = state.tolist()
    return repaired


def _arrays(records: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, ...]:
    def column(name: str, shape: Tuple[int, ...]) -> np.ndarray:
        return np.asarray([row[name] for row in records], dtype=np.float64).reshape((len(records),) + shape)

    return (
        column("left_transl_world", (3,)),
        column("right_transl_world", (3,)),
        column("left_orient_world", (3, 3)),
        column("right_orient_world", (3, 3)),
        column("left_hand_pose", (15, 3, 3)),
        column("right_hand_pose", (15, 3, 3)),
        np.asarray([row["state_mask"] for row in records], dtype=bool).reshape(-1, 2),
        column("extrinsics_w2c", (4, 4)),
    )


def _finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _full_episode_records(data_path: Path, episode: int) -> Tuple[List[Dict[str, Any]], pa.Schema]:
    parquet = pq.ParquetFile(data_path)
    table = parquet.read()
    if "episode_index" not in table.column_names:
        raise ValueError(f"repair preview 缺少 episode_index: {data_path}")
    episodes = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
    table = table.filter(pa.array(episodes == episode))
    records = table.to_pylist()
    records.sort(key=lambda row: int(row["frame_index"]))
    return records, table.schema


def _motion_fidelity(
    original: Sequence[Dict[str, Any]], repaired: Sequence[Dict[str, Any]], config: Dict[str, Any]
) -> Dict[str, Any]:
    old_arrays = _arrays(original)
    new_arrays = _arrays(repaired)
    masks = old_arrays[6]
    result: Dict[str, Any] = {}
    for side_index, side in enumerate(("left", "right")):
        old_position, new_position = old_arrays[side_index], new_arrays[side_index]
        old_wrist, new_wrist = old_arrays[side_index + 2], new_arrays[side_index + 2]
        old_pose, new_pose = old_arrays[side_index + 4], new_arrays[side_index + 4]
        valid = masks[:, side_index]
        pairs = valid[:-1] & valid[1:]

        def position_path(values: np.ndarray) -> float:
            return float(np.sum(np.linalg.norm(np.diff(values, axis=0), axis=-1)[pairs]))

        old_path, new_path = position_path(old_position), position_path(new_position)
        old_wrist_motion = float(np.sum(geodesic_degrees(old_wrist[:-1], old_wrist[1:])[pairs]))
        new_wrist_motion = float(np.sum(geodesic_degrees(new_wrist[:-1], new_wrist[1:])[pairs]))
        old_joint_motion = float(
            np.sum(np.mean(geodesic_degrees(old_pose[:-1], old_pose[1:]), axis=1)[pairs])
        )
        new_joint_motion = float(
            np.sum(np.mean(geodesic_degrees(new_pose[:-1], new_pose[1:]), axis=1)[pairs])
        )

        def ratio(after: float, before: float) -> Any:
            return None if before <= 1e-12 else after / before

        position_delta = np.linalg.norm(new_position - old_position, axis=1) * 1000.0
        wrist_delta = geodesic_degrees(old_wrist, new_wrist)
        joint_delta = np.max(geodesic_degrees(old_pose, new_pose), axis=1)
        result[side] = {
            "translation_path_retention": ratio(new_path, old_path),
            "wrist_motion_retention": ratio(new_wrist_motion, old_wrist_motion),
            "joint_motion_retention": ratio(new_joint_motion, old_joint_motion),
            "translation_correction_p99_mm": float(np.percentile(position_delta[valid], 99)) if np.any(valid) else None,
            "wrist_correction_p99_deg": float(np.percentile(wrist_delta[valid], 99)) if np.any(valid) else None,
            "joint_correction_p99_deg": float(np.percentile(joint_delta[valid], 99)) if np.any(valid) else None,
        }
    minimum = float(config.get("minimum_motion_retention", 0.65))
    maximum_translation = float(config.get("maximum_translation_correction_p99_mm", 30.0))
    maximum_rotation = float(config.get("maximum_rotation_correction_p99_deg", 8.0))
    retention_values = [
        value
        for side in ("left", "right")
        for key, value in result[side].items()
        if key.endswith("_retention") and value is not None
    ]
    result["minimum_motion_retention_required"] = minimum
    result["passes_motion_retention_gate"] = bool(retention_values) and min(retention_values) >= minimum
    result["maximum_translation_correction_p99_mm"] = maximum_translation
    result["maximum_rotation_correction_p99_deg"] = maximum_rotation
    result["passes_correction_gate"] = all(
        result[side]["translation_correction_p99_mm"] is not None
        and result[side]["translation_correction_p99_mm"] <= maximum_translation
        and result[side]["wrist_correction_p99_deg"] is not None
        and result[side]["wrist_correction_p99_deg"] <= maximum_rotation
        and result[side]["joint_correction_p99_deg"] is not None
        and result[side]["joint_correction_p99_deg"] <= maximum_rotation
        for side in ("left", "right")
    )
    result["passes_fidelity_gate"] = (
        result["passes_motion_retention_gate"] and result["passes_correction_gate"]
    )
    return result


def _source_motion_acceptance(metrics: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Strict vendor gate evaluated only on source metrics, never repaired data."""

    thresholds = config.get("thresholds", {})
    checks = [
        ("left_position_jitter_p99_mm", float(thresholds.get("position_jitter_warning_m", 0.008)) * 1000.0),
        ("right_position_jitter_p99_mm", float(thresholds.get("position_jitter_warning_m", 0.008)) * 1000.0),
        ("left_wrist_rotation_jitter_p99_deg", float(thresholds.get("rotation_jitter_warning_deg", 6.0))),
        ("right_wrist_rotation_jitter_p99_deg", float(thresholds.get("rotation_jitter_warning_deg", 6.0))),
        ("left_joint_rotation_jitter_p99_deg", float(thresholds.get("joint_jitter_warning_deg", 8.0))),
        ("right_joint_rotation_jitter_p99_deg", float(thresholds.get("joint_jitter_warning_deg", 8.0))),
        ("camera_translation_jitter_p99_mm", float(thresholds.get("camera_jitter_warning_m", 0.015)) * 1000.0),
        ("camera_rotation_jitter_p99_deg", float(thresholds.get("camera_rotation_jitter_warning_deg", 3.0))),
    ]
    failures: List[Dict[str, Any]] = []
    for key, limit in checks:
        value = metrics.get(key)
        if value is None or not np.isfinite(float(value)) or float(value) > limit:
            failures.append({"metric": key, "value": value, "maximum": limit})
    for side in ("left", "right"):
        spike_key = f"{side}_temporal_spike_count"
        spike_count = int(metrics.get(spike_key, 0) or 0)
        if spike_count > 0:
            failures.append({"metric": spike_key, "value": spike_count, "maximum": 0})
    return {
        "evaluated_on": "original_source_only",
        "repaired_preview_can_change_acceptance": False,
        "pass": not failures,
        "decision": "accept_motion_dimension" if not failures else "vendor_rework_required",
        "failures": failures,
    }


def write_repair_preview(
    dataset: Path,
    episode: int,
    output: Path,
    config: Dict[str, Any],
    mano_renderer: Optional[ManoOverlayRenderer] = None,
    video_key: str = "observation.images.ego",
    start_frame: int = 0,
    max_frames: Optional[int] = None,
) -> Dict[str, Any]:
    dataset = dataset.expanduser().resolve()
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    route = _episode_route(dataset, episode, video_key)
    info = json.loads((dataset / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = _fps(info, video_key)
    original, source_schema = _full_episode_records(route["data_path"], episode)
    repair_config = config.get("repair", {})
    repaired = repair_episode_records(original, fps, repair_config)

    source_timestamps = np.asarray(
        [row["timestamp"] for row in original], dtype=np.float64
    )
    before, _, _, _ = analyze_temporal_quality(
        *_arrays(original),
        fps,
        config,
        episode,
        str(route["data_path"]),
        timestamps=source_timestamps,
    )
    after, _, _, _ = analyze_temporal_quality(
        *_arrays(repaired),
        fps,
        config,
        episode,
        str(route["data_path"]),
        timestamps=source_timestamps,
    )
    keys = [key for key in before if "jitter_p99" in key]
    reductions = {}
    for key in keys:
        prior, current = float(before[key]), float(after[key])
        reductions[key] = None if not np.isfinite(prior) or prior == 0 else 1.0 - current / prior

    preview_path = output / "repair-preview.parquet"
    pq.write_table(
        pa.Table.from_pylist(repaired, schema=source_schema),
        preview_path,
        compression="zstd",
    )
    delta_rows = []
    for old, new in zip(original, repaired):
        row: Dict[str, Any] = {"frame_index": int(old["frame_index"]), "state_mask": old["state_mask"]}
        for side in ("left", "right"):
            old_position = np.asarray(old[f"{side}_transl_world"], dtype=np.float64)
            new_position = np.asarray(new[f"{side}_transl_world"], dtype=np.float64)
            row[f"{side}_translation_delta_mm"] = float(np.linalg.norm(new_position - old_position) * 1000.0)
        delta_rows.append(row)
    delta_path = output / "repair-deltas.parquet"
    pq.write_table(pa.Table.from_pylist(delta_rows), delta_path, compression="zstd")

    summary: Dict[str, Any] = {
        "mode": "derived_preview_only",
        "source_unchanged": True,
        "dataset": str(dataset),
        "episode_index": episode,
        "frames": len(repaired),
        "fps": fps,
        "source_parquet": str(route["data_path"]),
        "repair_preview": str(preview_path),
        "repair_deltas": str(delta_path),
        "config": repair_config,
        "before": before,
        "after": after,
        "source_motion_acceptance": _source_motion_acceptance(before, config),
        "relative_jitter_reduction": reductions,
        "motion_fidelity": _motion_fidelity(original, repaired, repair_config),
    }
    summary["recommendation"] = summary["source_motion_acceptance"]["decision"]
    if mano_renderer is not None:
        video_path = output / f"episode-{episode:06d}-repaired-annotated.mp4"
        summary["annotated_video"] = render_annotated_episode(
            dataset,
            episode,
            video_path,
            mano_renderer,
            video_key=video_key,
            start_frame=start_frame,
            max_frames=max_frames,
            records_override=repaired,
        )
    summary = _finite_json(summary)
    metrics_path = output / "repair-metrics.json"
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["metrics"] = str(metrics_path)
    return summary
