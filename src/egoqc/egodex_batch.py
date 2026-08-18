from __future__ import annotations

import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from .adapters import EgoDexHDF5Adapter
from .report import write_json, write_jsonl


DEFAULT_PARTITIONS = ("extra", "part1", "part2", "part3", "part4", "part5")


def _stable_rank(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def select_egodex_episodes(
    dataset: Path,
    *,
    episodes_per_task: int = 2,
    seed: int = 17,
    partitions: Sequence[str] = DEFAULT_PARTITIONS,
) -> List[Dict[str, Any]]:
    """Select a reproducible, task-balanced pilot without recursively walking NFS."""
    if episodes_per_task <= 0:
        raise ValueError("episodes_per_task 必须大于 0")
    dataset = dataset.expanduser().resolve()
    selected: List[Dict[str, Any]] = []
    for partition in partitions:
        partition_root = dataset / partition
        if not partition_root.is_dir():
            continue
        for task_root in sorted(path for path in partition_root.iterdir() if path.is_dir()):
            candidates = sorted(
                (
                    path
                    for path in task_root.iterdir()
                    if path.is_file() and path.suffix.lower() in {".hdf5", ".h5"}
                ),
                key=lambda path: _stable_rank(path.relative_to(dataset).as_posix(), seed),
            )
            for hdf5_path in candidates[:episodes_per_task]:
                relative = hdf5_path.relative_to(dataset)
                selected.append(
                    {
                        "partition": partition,
                        "task": task_root.name,
                        "episode_id": relative.with_suffix("").as_posix(),
                        "hdf5_path": str(hdf5_path),
                        "video_path": str(hdf5_path.with_suffix(".mp4")),
                    }
                )
    return selected


def _longest_run(mask: np.ndarray, value: bool) -> int:
    longest = current = 0
    for item in mask:
        if bool(item) is value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _eligible_visible_frames(mask: np.ndarray, minimum_frames: int) -> int:
    total = current = 0
    for item in np.append(mask, False):
        if bool(item):
            current += 1
        else:
            if current >= minimum_frames:
                total += current
            current = 0
    return total


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _profile_one(
    dataset: Path,
    selected: Dict[str, Any],
    *,
    confidence_threshold: float,
    minimum_duration_s: float,
    minimum_fps: float,
    minimum_width: int,
    minimum_height: int,
    maximum_absence_s: float,
) -> Dict[str, Any]:
    canonical = EgoDexHDF5Adapter().load_episode(
        dataset,
        selected["episode_id"],
        confidence_threshold=confidence_threshold,
    )
    summary = canonical.summary()
    fps = float(canonical.video.fps)
    left = canonical.hands["left"]
    right = canonical.hands["right"]
    any_visible = np.asarray(left.valid | right.valid, dtype=bool)
    longest_visible_s = _longest_run(any_visible, True) / fps
    longest_absence_s = _longest_run(any_visible, False) / fps
    minimum_visible_frames = int(math.ceil(minimum_duration_s * fps))
    eligible_visible_s = _eligible_visible_frames(any_visible, minimum_visible_frames) / fps

    gates = {
        "duration_at_least_minimum": summary["duration_s"] >= minimum_duration_s,
        "fps_at_least_minimum": fps >= minimum_fps,
        "resolution_at_least_720p": (
            canonical.video.width >= minimum_width
            and canonical.video.height >= minimum_height
        ),
        "continuous_hand_visibility_at_least_minimum": (
            longest_visible_s >= minimum_duration_s
        ),
        "hand_absence_not_over_limit": longest_absence_s <= maximum_absence_s,
        "no_audio": canonical.video.audio_streams == 0,
    }
    hard_pass = all(gates.values())

    hand_summaries = summary["hands"]
    active_hands = [
        hand for hand in hand_summaries.values() if float(hand["valid_ratio"]) >= 0.05
    ]
    if not active_hands:
        active_hands = list(hand_summaries.values())
    joint_ratio = float(np.mean([
        hand["joint_values_confident_ratio"] or 0.0 for hand in active_hands
    ]))
    all_joint_ratio = float(np.mean([
        hand["all_joints_confident_ratio"] or 0.0 for hand in active_hands
    ]))
    confidence_mean = float(np.mean([
        hand["joint_confidence_mean"] or 0.0 for hand in active_hands
    ]))
    confidence_p05 = float(np.mean([
        hand["joint_confidence_p05"] or 0.0 for hand in active_hands
    ]))
    visible_ratio = float(np.mean(any_visible)) if any_visible.size else 0.0
    # This is only a within-batch ranking signal. It is not an accuracy estimate.
    annotation_score = float(np.clip(
        0.30 * visible_ratio
        + 0.30 * joint_ratio
        + 0.20 * all_joint_ratio
        + 0.10 * confidence_mean
        + 0.10 * confidence_p05,
        0.0,
        1.0,
    ))
    return {
        **selected,
        "source_format": canonical.source_format,
        "source_readonly": True,
        "labels": summary["labels"],
        "capabilities": summary["capabilities"],
        "video": summary["video"],
        "duration_s": summary["duration_s"],
        "hand_metrics": {
            "any_hand_visible_ratio": visible_ratio,
            "longest_visible_s": longest_visible_s,
            "longest_absence_s": longest_absence_s,
            "eligible_visible_s": eligible_visible_s,
            "left": hand_summaries["left"],
            "right": hand_summaries["right"],
        },
        "hard_gates": gates,
        "hard_pass": hard_pass,
        "annotation_score": annotation_score,
        "annotation_score_semantics": (
            "weak within-batch ranking signal; not ground-truth accuracy or acceptance proof"
        ),
        "provenance": summary["provenance"],
    }


def _assign_tiers(
    profiles: List[Dict[str, Any]],
    *,
    clean_quantile: float,
    hard_negative_quantile: float,
) -> Dict[str, float | None]:
    passing_scores = np.asarray(
        [row["annotation_score"] for row in profiles if row["hard_pass"]],
        dtype=np.float64,
    )
    clean_threshold = (
        float(np.quantile(passing_scores, clean_quantile)) if passing_scores.size else None
    )
    hard_negative_threshold = (
        float(np.quantile(passing_scores, hard_negative_quantile))
        if passing_scores.size else None
    )
    for row in profiles:
        if not row["hard_pass"]:
            tier = "programmatic-reject"
            reason = "hard_gate_failed"
        elif clean_threshold is not None and row["annotation_score"] >= clean_threshold:
            tier = "candidate-clean"
            reason = "batch_quality_upper_quantile"
        elif (
            hard_negative_threshold is not None
            and row["annotation_score"] <= hard_negative_threshold
        ):
            tier = "hard-negative"
            reason = "batch_quality_lower_quantile"
        else:
            tier = "review"
            reason = "human_review_required"
        row["candidate_tier"] = tier
        row["tier_reason"] = reason
        row["label_status"] = "weak_candidate_not_gold"
        row["visual_model_eligible"] = tier in {
            "candidate-clean", "review", "hard-negative"
        }
    return {
        "candidate_clean_min_score": clean_threshold,
        "hard_negative_max_score": hard_negative_threshold,
    }


def _counts(rows: Iterable[Dict[str, Any]], key: str) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))


def build_egodex_training_candidates(
    dataset: Path,
    output: Path,
    *,
    episodes_per_task: int = 2,
    seed: int = 17,
    workers: int = 8,
    confidence_threshold: float = 0.5,
    minimum_duration_s: float = 5.0,
    minimum_fps: float = 29.97,
    minimum_width: int = 1280,
    minimum_height: int = 720,
    maximum_absence_s: float = 1.0,
    clean_quantile: float = 0.67,
    hard_negative_quantile: float = 0.20,
    partitions: Sequence[str] = DEFAULT_PARTITIONS,
    resume: bool = False,
) -> Dict[str, Any]:
    """Build read-only EgoDex QC candidates and preserve every failure as evidence."""
    if workers <= 0:
        raise ValueError("workers 必须大于 0")
    if not 0.0 <= hard_negative_quantile < clean_quantile <= 1.0:
        raise ValueError("质量分位需满足 0 <= hard-negative < clean <= 1")
    dataset = dataset.expanduser().resolve()
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = select_egodex_episodes(
        dataset,
        episodes_per_task=episodes_per_task,
        seed=seed,
        partitions=partitions,
    )
    profiles: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    if resume:
        for path, target in (
            (output / "profiles.jsonl", profiles),
            (output / "errors.jsonl", errors),
        ):
            if path.exists():
                target.extend(
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
    completed_ids = {
        str(row["episode_id"])
        for row in [*profiles, *errors]
    }
    pending = [row for row in selected if row["episode_id"] not in completed_ids]
    kwargs = {
        "confidence_threshold": confidence_threshold,
        "minimum_duration_s": minimum_duration_s,
        "minimum_fps": minimum_fps,
        "minimum_width": minimum_width,
        "minimum_height": minimum_height,
        "maximum_absence_s": maximum_absence_s,
    }
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_profile_one, dataset, row, **kwargs): row
            for row in pending
        }
        for future in as_completed(futures):
            source = futures[future]
            try:
                profiles.append(future.result())
            except Exception as error:  # preserve a bad source as a reviewable result
                errors.append({
                    **source,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "candidate_tier": "technical-blocked",
                    "tier_reason": "adapter_or_integrity_failure",
                    "label_status": "weak_candidate_not_gold",
                    "visual_model_eligible": False,
                })
    profiles.sort(key=lambda row: row["episode_id"])
    errors.sort(key=lambda row: row["episode_id"])
    for row in errors:
        row.update({
            "candidate_tier": "technical-blocked",
            "tier_reason": "adapter_or_integrity_failure",
            "label_status": "weak_candidate_not_gold",
            "visual_model_eligible": False,
        })
    thresholds = _assign_tiers(
        profiles,
        clean_quantile=clean_quantile,
        hard_negative_quantile=hard_negative_quantile,
    )
    tier_rows = {
        name: [row for row in profiles if row["candidate_tier"] == name]
        for name in (
            "candidate-clean", "review", "hard-negative", "programmatic-reject"
        )
    }
    tier_rows["technical-blocked"] = list(errors)
    write_jsonl(output / "selected.jsonl", selected)
    write_jsonl(output / "profiles.jsonl", profiles)
    write_jsonl(output / "errors.jsonl", errors)
    for name, rows in tier_rows.items():
        write_jsonl(output / f"{name}.jsonl", rows)
    summary = {
        "schema_version": "egoqc-egodex-training-candidates-v1",
        "dataset": str(dataset),
        "output": str(output),
        "source_readonly": True,
        "selection": {
            "seed": seed,
            "episodes_per_task": episodes_per_task,
            "partitions": list(partitions),
            "selected": len(selected),
            "reused": len(selected) - len(pending),
            "processed_this_run": len(pending),
            "tasks": len({(row["partition"], row["task"]) for row in selected}),
        },
        "profiled": len(profiles),
        "errors": len(errors),
        "tier_counts": {name: len(rows) for name, rows in tier_rows.items()},
        "partition_counts": _counts(selected, "partition"),
        "hard_gate_pass": sum(bool(row["hard_pass"]) for row in profiles),
        "thresholds": {
            "confidence": confidence_threshold,
            "minimum_duration_s": minimum_duration_s,
            "minimum_fps": minimum_fps,
            "minimum_width": minimum_width,
            "minimum_height": minimum_height,
            "maximum_absence_s": maximum_absence_s,
            "clean_quantile": clean_quantile,
            "hard_negative_quantile": hard_negative_quantile,
            **thresholds,
        },
        "label_policy": {
            "status": "weak_candidate_not_gold",
            "candidate_clean_requires_human_audit_before_gold": True,
            "validation_and_test_require_human_gold": True,
        },
        "artifacts": {
            "selected": str(output / "selected.jsonl"),
            "profiles": str(output / "profiles.jsonl"),
            "candidate_clean": str(output / "candidate-clean.jsonl"),
            "review": str(output / "review.jsonl"),
            "hard_negative": str(output / "hard-negative.jsonl"),
            "programmatic_reject": str(output / "programmatic-reject.jsonl"),
            "technical_blocked": str(output / "technical-blocked.jsonl"),
            "errors": str(output / "errors.jsonl"),
        },
    }
    summary = {key: _finite_or_none(value) for key, value in summary.items()}
    write_json(output / "summary.json", summary)
    return summary
