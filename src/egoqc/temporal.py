from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .math3d import geodesic_degrees
from .types import Issue


def _percentile(values: np.ndarray, q: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, q)) if finite.size else float("nan")


def _nanmax_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.full(len(values), np.nan, dtype=np.float64)
    valid = np.isfinite(values).any(axis=tuple(range(1, values.ndim)))
    if np.any(valid):
        result[valid] = np.max(
            np.where(np.isfinite(values[valid]), values[valid], -np.inf),
            axis=tuple(range(1, values.ndim)),
        )
    return result


def _severity(value: float, warning: float, error: float) -> str:
    if not np.isfinite(value) or value <= warning:
        return "info"
    return "error" if value > error else "warning"


def _position_residual(points: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Distance to the local linear interpolation, in metres."""

    points = np.asarray(points, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    residual = np.full(len(points), np.nan, dtype=np.float64)
    if len(points) < 3:
        return residual
    centres = valid[:-2] & valid[1:-1] & valid[2:]
    values = np.linalg.norm(points[1:-1] - (points[:-2] + points[2:]) / 2.0, axis=1)
    residual[1:-1] = np.where(centres, values, np.nan)
    return residual


def _project_to_so3(matrices: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(np.asarray(matrices, dtype=np.float64))
    rotations = u @ vh
    negative = np.linalg.det(rotations) < 0
    if np.any(negative):
        u = u.copy()
        u[negative, :, -1] *= -1
        rotations = u @ vh
    return rotations


def _rotation_residual(rotations: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """SO(3) residual to the projected neighbour midpoint, in degrees."""

    rotations = np.asarray(rotations, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    residual = np.full(rotations.shape[:-2], np.nan, dtype=np.float64)
    if len(rotations) < 3:
        return residual
    midpoint = _project_to_so3(rotations[:-2] + rotations[2:])
    values = geodesic_degrees(midpoint, rotations[1:-1])
    centres = valid[:-2] & valid[1:-1] & valid[2:]
    if values.ndim > 1:
        centres = centres.reshape((-1,) + (1,) * (values.ndim - 1))
    residual[1:-1] = np.where(centres, values, np.nan)
    return residual


def _robust_outliers(values: np.ndarray, floor: float, z_limit: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return np.zeros(values.shape, dtype=bool)
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    robust_scale = max(1.4826 * mad, floor / max(z_limit, 1.0), 1e-12)
    return np.isfinite(values) & (values > floor) & ((values - median) / robust_scale > z_limit)


def _false_runs(mask: np.ndarray, maximum: int) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    runs: List[Tuple[int, int]] = []
    start = None
    for index, value in enumerate(mask):
        if not value and start is None:
            start = index
        if value and start is not None:
            if start > 0 and index - start <= maximum:
                runs.append((start, index - 1))
            start = None
    return runs


def _longest_true_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in np.asarray(values, dtype=bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return int(longest)


def _velocity_statistics(
    points: np.ndarray,
    valid: np.ndarray,
    timestamps: np.ndarray,
) -> Tuple[np.ndarray, float, float, float]:
    """Return per-frame segment speeds and the contractual median+3*MAD limit."""

    points = np.asarray(points, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    speeds = np.full(len(points), np.nan, dtype=np.float64)
    if len(points) < 2:
        return speeds, float("nan"), float("nan"), float("nan")
    dt = np.diff(timestamps)
    pair_valid = (
        valid[:-1]
        & valid[1:]
        & np.isfinite(points[:-1]).all(axis=1)
        & np.isfinite(points[1:]).all(axis=1)
        & np.isfinite(dt)
        & (dt > 0)
    )
    step = np.linalg.norm(np.diff(points, axis=0), axis=1)
    speeds[1:] = np.where(pair_valid, step / np.where(dt > 0, dt, np.nan), np.nan)
    finite = speeds[np.isfinite(speeds)]
    if not finite.size:
        return speeds, float("nan"), float("nan"), float("nan")
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    return speeds, median, mad, median + 3.0 * mad


def _bad_frame(
    frame_index: int,
    code: str,
    severity: str,
    file_name: str,
    *,
    side: str | None = None,
    measured: float | None = None,
    threshold: float | None = None,
    unit: str | None = None,
) -> Dict[str, Any]:
    return {
        "frame_index": int(frame_index),
        "code": code,
        "severity": severity,
        "side": side,
        "measured": measured,
        "threshold": threshold,
        "unit": unit,
        "file": file_name,
    }


def analyze_temporal_quality(
    left_world: np.ndarray,
    right_world: np.ndarray,
    left_orient: np.ndarray,
    right_orient: np.ndarray,
    left_pose: np.ndarray,
    right_pose: np.ndarray,
    state_mask: np.ndarray,
    extrinsics_w2c: np.ndarray,
    fps: float,
    config: Dict[str, Any],
    episode_index: int,
    file_name: str,
    timestamps: np.ndarray | None = None,
) -> Tuple[Dict[str, Any], List[Issue], List[int], List[Dict[str, Any]]]:
    """Cheap temporal QC over Parquet arrays; never decodes video."""

    thresholds = config.get("thresholds", {})
    temporal = config.get("temporal", {})
    z_limit = float(temporal.get("robust_z", 8.0))
    max_candidates = int(temporal.get("max_candidate_frames", 6))
    metrics: Dict[str, Any] = {}
    issues: List[Issue] = []
    candidate_scores: Dict[int, float] = {}
    bad_frames: List[Dict[str, Any]] = []
    if timestamps is None:
        timestamps = np.arange(len(left_world), dtype=np.float64) / fps
    else:
        timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)

    sides = (
        ("left", left_world, left_orient, left_pose, np.asarray(state_mask)[:, 0]),
        ("right", right_world, right_orient, right_pose, np.asarray(state_mask)[:, 1]),
    )
    for side, position, orient, pose, valid in sides:
        position_residual = _position_residual(position, valid)
        wrist_rotation_residual = _rotation_residual(orient, valid)
        joint_rotation_residual = _rotation_residual(pose, valid)
        joint_frame_residual = _nanmax_rows(joint_rotation_residual)

        position_p99 = _percentile(position_residual, 99)
        wrist_rotation_p99 = _percentile(wrist_rotation_residual, 99)
        joint_rotation_p99 = _percentile(joint_frame_residual, 99)
        metrics[f"{side}_position_jitter_p99_mm"] = position_p99 * 1000.0
        metrics[f"{side}_wrist_rotation_jitter_p99_deg"] = wrist_rotation_p99
        metrics[f"{side}_joint_rotation_jitter_p99_deg"] = joint_rotation_p99

        checks = (
            (
                "position_jitter",
                position_p99,
                float(thresholds.get("position_jitter_warning_m", 0.008)),
                float(thresholds.get("position_jitter_error_m", 0.020)),
                f"{side} wrist 局部位置残差 p99={position_p99 * 1000.0:.2f}mm",
            ),
            (
                "wrist_rotation_jitter",
                wrist_rotation_p99,
                float(thresholds.get("rotation_jitter_warning_deg", 6.0)),
                float(thresholds.get("rotation_jitter_error_deg", 15.0)),
                f"{side} wrist SO(3) 局部残差 p99={wrist_rotation_p99:.2f}°",
            ),
            (
                "joint_rotation_jitter",
                joint_rotation_p99,
                float(thresholds.get("joint_jitter_warning_deg", 8.0)),
                float(thresholds.get("joint_jitter_error_deg", 20.0)),
                f"{side} joint SO(3) 局部残差 p99={joint_rotation_p99:.2f}°",
            ),
        )
        for code, value, warning, error, message in checks:
            severity = _severity(value, warning, error)
            if severity != "info":
                issues.append(Issue(code, severity, message, episode_index, file_name))

        position_spikes = _robust_outliers(
            position_residual,
            float(thresholds.get("position_spike_m", 0.025)),
            z_limit,
        )
        rotation_spikes = _robust_outliers(
            np.maximum(wrist_rotation_residual, joint_frame_residual),
            float(thresholds.get("rotation_spike_deg", 20.0)),
            z_limit,
        )
        spike_frames = np.flatnonzero(position_spikes | rotation_spikes)
        metrics[f"{side}_temporal_spike_count"] = int(len(spike_frames))
        if len(spike_frames):
            frames = [int(value) for value in spike_frames[:32]]
            issues.append(
                Issue(
                    "temporal_spike",
                    "error",
                    f"{side} 检测到 {len(spike_frames)} 个孤立跳点",
                    episode_index,
                    file_name,
                    {"frames": frames},
                )
            )
            for frame in spike_frames:
                bad_frames.append(
                    _bad_frame(
                        int(frame),
                        "temporal_spike",
                        "error",
                        file_name,
                        side=side,
                    )
                )
                candidate_scores[int(frame)] = max(
                    candidate_scores.get(int(frame), 0.0),
                    float(np.nan_to_num(position_residual[frame]) * 1000.0)
                    + float(np.nan_to_num(np.maximum(wrist_rotation_residual, joint_frame_residual)[frame])),
                )

        flickers = _false_runs(valid, int(temporal.get("mask_flicker_max_frames", 3)))
        metrics[f"{side}_mask_flicker_count"] = len(flickers)
        if flickers:
            frames = [start for start, _ in flickers]
            issues.append(
                Issue(
                    "mask_flicker",
                    "warning",
                    f"{side} 检测到 {len(flickers)} 个短时 mask 缺失段",
                    episode_index,
                    file_name,
                    {"segments": [[start, end] for start, end in flickers[:32]]},
                )
            )
            for frame in frames:
                candidate_scores[frame] = max(candidate_scores.get(frame, 0.0), 1000.0)
            for start, end in flickers:
                for frame in range(start, end + 1):
                    bad_frames.append(
                        _bad_frame(
                            frame,
                            "mask_flicker",
                            "warning",
                            file_name,
                            side=side,
                        )
                    )

        speeds, speed_median, speed_mad, speed_limit = _velocity_statistics(
            position, valid, timestamps
        )
        metrics[f"{side}_velocity_median_m_s"] = speed_median
        metrics[f"{side}_velocity_mad_m_s"] = speed_mad
        metrics[f"{side}_velocity_limit_m_s"] = speed_limit
        velocity_outliers = (
            np.isfinite(speeds)
            & np.isfinite(speed_limit)
            & (speeds > speed_limit + max(1e-12, abs(speed_limit) * 1e-12))
        )
        velocity_frames = np.flatnonzero(velocity_outliers)
        metrics[f"{side}_velocity_outlier_count"] = int(len(velocity_frames))
        metrics[f"{side}_velocity_outlier_ratio"] = float(np.mean(velocity_outliers))
        if len(velocity_frames):
            issues.append(
                Issue(
                    "instantaneous_velocity_outlier",
                    "error",
                    f"{side} 有 {len(velocity_frames)} 帧速度不满足 median(V)+3×MAD",
                    episode_index,
                    file_name,
                    {
                        "frames": [int(value) for value in velocity_frames[:32]],
                        "median_m_s": speed_median,
                        "mad_m_s": speed_mad,
                        "limit_m_s": speed_limit,
                    },
                )
            )
            for frame in velocity_frames:
                bad_frames.append(
                    _bad_frame(
                        int(frame),
                        "instantaneous_velocity_outlier",
                        "error",
                        file_name,
                        side=side,
                        measured=float(speeds[frame]),
                        threshold=float(speed_limit),
                        unit="m/s",
                    )
                )
                candidate_scores[int(frame)] = max(
                    candidate_scores.get(int(frame), 0.0),
                    float(speeds[frame]),
                )

        if len(position) > 1:
            translation_step = np.linalg.norm(np.diff(position, axis=0), axis=1)
            pose_step = _nanmax_rows(geodesic_degrees(pose[:-1], pose[1:]))
            pair_valid = valid[:-1] & valid[1:]
            stationary = pair_valid & (
                translation_step < float(temporal.get("freeze_position_step_m", 0.0002))
            ) & (pose_step < float(temporal.get("freeze_joint_step_deg", 0.2)))
            freeze_frames = _longest_true_run(stationary) + (1 if np.any(stationary) else 0)
        else:
            freeze_frames = 0
        metrics[f"{side}_pose_freeze_longest_frames"] = freeze_frames
        freeze_warning = int(temporal.get("freeze_warning_frames", max(90, round(fps * 3))))
        if freeze_frames >= freeze_warning:
            issues.append(
                Issue(
                    "pose_freeze_candidate",
                    "info",
                    f"{side} 最长近静止段 {freeze_frames} 帧，需结合视频确认",
                    episode_index,
                    file_name,
                )
            )

    camera_valid = np.isfinite(extrinsics_w2c).all(axis=(1, 2))
    camera_position_residual = _position_residual(extrinsics_w2c[:, :3, 3], camera_valid)
    camera_rotation_residual = _rotation_residual(extrinsics_w2c[:, :3, :3], camera_valid)
    camera_position_p99 = _percentile(camera_position_residual, 99)
    camera_rotation_p99 = _percentile(camera_rotation_residual, 99)
    metrics["camera_translation_jitter_p99_mm"] = camera_position_p99 * 1000.0
    metrics["camera_rotation_jitter_p99_deg"] = camera_rotation_p99
    camera_warning_m = float(thresholds.get("camera_jitter_warning_m", 0.015))
    camera_error_m = float(thresholds.get("camera_jitter_error_m", 0.030))
    camera_warning_deg = float(thresholds.get("camera_rotation_jitter_warning_deg", 3.0))
    camera_error_deg = float(thresholds.get("camera_rotation_jitter_error_deg", 8.0))
    camera_severity = "info"
    if camera_position_p99 > camera_error_m or camera_rotation_p99 > camera_error_deg:
        camera_severity = "error"
    elif camera_position_p99 > camera_warning_m or camera_rotation_p99 > camera_warning_deg:
        camera_severity = "warning"
    if camera_severity != "info":
        issues.append(
            Issue(
                "camera_jitter",
                camera_severity,
                f"相机局部残差 p99={camera_position_p99 * 1000.0:.2f}mm / {camera_rotation_p99:.2f}°",
                episode_index,
                file_name,
            )
        )

    ranked = sorted(candidate_scores, key=lambda frame: candidate_scores[frame], reverse=True)
    return metrics, issues, ranked[:max_candidates], bad_frames
