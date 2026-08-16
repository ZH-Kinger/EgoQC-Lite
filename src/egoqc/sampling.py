from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np


def _top_indices(values: np.ndarray, count: int) -> Iterable[int]:
    if count <= 0 or len(values) == 0:
        return []
    finite = np.nan_to_num(values, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    count = min(count, len(values))
    return np.argpartition(finite, -count)[-count:].tolist()


def select_sample_frames(
    length: int,
    config: Dict[str, Any],
    left_wrist: np.ndarray,
    right_wrist: np.ndarray,
    state_mask: np.ndarray,
    fps: float,
    episode_index: int,
    priority_frames: Iterable[int] = (),
) -> List[int]:
    if length <= 0:
        return []
    selected = {
        int(round(float(q) * (length - 1)))
        for q in config.get("quantiles", [0.0, 0.5, 1.0])
    }
    velocity = np.zeros(length, dtype=np.float64)
    for wrist in (left_wrist, right_wrist):
        if len(wrist) == length and length > 2:
            acceleration = np.diff(wrist, n=2, axis=0) * (fps**2)
            magnitude = np.linalg.norm(acceleration, axis=1)
            velocity[2:] = np.maximum(velocity[2:], magnitude)
    selected.update(_top_indices(velocity, int(config.get("kinematic_hotspots", 3))))

    if len(state_mask) == length and length > 1:
        transitions = np.flatnonzero(np.any(state_mask[1:] != state_mask[:-1], axis=1)) + 1
        limit = int(config.get("mask_transitions", 2))
        selected.update(transitions[:limit].tolist())

    random_count = int(config.get("random_frames", 1))
    rng = np.random.default_rng(int(config.get("seed", 17)) + episode_index)
    if random_count > 0:
        selected.update(rng.integers(0, length, size=min(random_count, length)).tolist())

    maximum = int(config.get("max_frames_per_episode", 12))
    priority = list(dict.fromkeys(
        int(frame) for frame in priority_frames if 0 <= int(frame) < length
    ))
    remaining = sorted(selected - set(priority))
    return sorted((priority + remaining[: max(0, maximum - len(priority))])[:maximum])
