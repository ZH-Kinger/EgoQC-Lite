from __future__ import annotations

import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .adapters import EgoDexHDF5Adapter, _json_safe
from .report import write_json, write_jsonl


DEFAULT_PARTITIONS = ("extra", "part1", "part2", "part3", "part4", "part5")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def ensure_readonly_source_boundary(dataset: Path, output: Path) -> None:
    """Refuse derived writes inside a source tree or the deployed raw-data mount."""
    dataset = dataset.expanduser().resolve()
    output = output.expanduser().resolve()
    if _is_within(output, dataset):
        raise ValueError(f"输出目录不能位于只读源数据集内: {output}")
    raw_mount = Path("/mnt/data")
    if _is_within(dataset, raw_mount) and _is_within(output, raw_mount):
        raise ValueError(
            f"源数据位于受保护挂载 {raw_mount}；派生产物必须写到 /mnt/workspace 等独立目录"
        )


def _stable_rank(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def select_egodex_episodes(
    dataset: Path,
    *,
    episodes_per_task: int = 2,
    seed: int = 17,
    partitions: Sequence[str] = DEFAULT_PARTITIONS,
    inventory_cache: Optional[Path] = None,
    refresh_inventory: bool = False,
) -> List[Dict[str, Any]]:
    """Select a reproducible, task-balanced pilot without recursively walking NFS."""
    if episodes_per_task <= 0:
        raise ValueError("episodes_per_task 必须大于 0")
    dataset = dataset.expanduser().resolve()
    inventory_cache = inventory_cache.expanduser().resolve() if inventory_cache else None
    inventory: List[Dict[str, Any]] = []
    if inventory_cache and inventory_cache.is_file() and not refresh_inventory:
        inventory = [
            json.loads(line)
            for line in inventory_cache.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        roots = {str(row.get("dataset")) for row in inventory}
        if roots != {str(dataset)}:
            raise ValueError(
                f"inventory 数据集不匹配: cache={sorted(roots)}, requested={dataset}"
            )
    else:
        for partition in partitions:
            partition_root = dataset / partition
            if not partition_root.is_dir():
                continue
            for task_root in sorted(path for path in partition_root.iterdir() if path.is_dir()):
                for hdf5_path in task_root.iterdir():
                    if not hdf5_path.is_file() or hdf5_path.suffix.lower() not in {".hdf5", ".h5"}:
                        continue
                    relative = hdf5_path.relative_to(dataset)
                    inventory.append({
                        "schema_version": "egoqc-egodex-inventory-v1",
                        "dataset": str(dataset),
                        "partition": partition,
                        "task": task_root.name,
                        "episode_id": relative.with_suffix("").as_posix(),
                        "hdf5_path": str(hdf5_path),
                        "video_path": str(hdf5_path.with_suffix(".mp4")),
                    })
        inventory.sort(key=lambda row: row["episode_id"])
        if inventory_cache:
            write_jsonl(inventory_cache, inventory)

    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    allowed = set(partitions)
    for row in inventory:
        if row["partition"] in allowed:
            grouped.setdefault((row["partition"], row["task"]), []).append(row)
    selected: List[Dict[str, Any]] = []
    for key in sorted(grouped):
        candidates = sorted(
            grouped[key], key=lambda row: _stable_rank(row["episode_id"], seed)
        )
        selected.extend(candidates[:episodes_per_task])
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


def _egodex_joint_names(side: str) -> List[str]:
    prefix = side.lower()
    names = [f"{prefix}Hand"]
    names.extend(
        f"{prefix}Thumb{segment}"
        for segment in ("Knuckle", "IntermediateBase", "IntermediateTip", "Tip")
    )
    names.extend(
        f"{prefix}{finger}Finger{segment}"
        for finger in ("Index", "Middle", "Ring", "Little")
        for segment in (
            "Metacarpal", "Knuckle", "IntermediateBase", "IntermediateTip", "Tip"
        )
    )
    return names


def _confidence_summary(
    side: str, confidences: np.ndarray, threshold: float
) -> Dict[str, Any]:
    root = confidences[:, 0]
    finite = confidences[np.isfinite(confidences)]
    valid = np.isfinite(root) & (root >= threshold)
    return {
        "side": side,
        "joint_count": int(confidences.shape[1]),
        "joint_names": _egodex_joint_names(side),
        "valid_frames": int(np.count_nonzero(valid)),
        "valid_ratio": float(np.mean(valid)) if len(valid) else 0.0,
        "valid_semantics": "root_presence_above_confidence_threshold",
        "finite_frames": None,
        "root_confidence_mean": float(np.mean(root)) if root.size else None,
        "joint_confidence_mean": float(np.mean(finite)) if finite.size else None,
        "joint_confidence_p05": float(np.quantile(finite, 0.05)) if finite.size else None,
        "confidence_threshold": threshold,
        "all_joints_confident_ratio": float(
            np.mean(np.all(confidences >= threshold, axis=1))
        ) if confidences.size else None,
        "joint_values_confident_ratio": float(np.mean(confidences >= threshold))
        if confidences.size else None,
        "local_origin": "wrist_root_transform; non-MANO source joint hierarchy",
        "source_model": "EgoDex joint SE(3)",
        "profile_mode": "confidence_only",
    }


def _profile_one_fast(
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
    """Profile an episode without loading joint/camera 4x4 transform tensors."""
    try:
        import av
        import h5py
    except ImportError as error:
        raise RuntimeError("快速 EgoDex 画像需要 h5py 和 PyAV") from error

    hdf5_path = Path(selected["hdf5_path"])
    video_path = Path(selected["video_path"])
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    with h5py.File(hdf5_path, "r") as handle:
        required = ("camera/intrinsic", "transforms/camera", "transforms", "confidences")
        missing = [name for name in required if name not in handle]
        if missing:
            raise ValueError(f"EgoDex HDF5 缺少字段: {missing}")
        frame_count = int(handle["transforms/camera"].shape[0])
        hand_confidences: Dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            names = _egodex_joint_names(side)
            missing_transforms = [name for name in names if name not in handle["transforms"]]
            missing_confidences = [name for name in names if name not in handle["confidences"]]
            if missing_transforms:
                raise ValueError(f"{side} 缺少 transform: {missing_transforms}")
            if missing_confidences:
                raise ValueError(f"{side} 缺少 confidence: {missing_confidences}")
            values = np.stack(
                [np.asarray(handle["confidences"][name], dtype=np.float64) for name in names],
                axis=1,
            )
            if values.shape[0] != frame_count:
                raise ValueError(
                    f"{side} confidence 帧数={values.shape[0]}，camera 帧数={frame_count}"
                )
            hand_confidences[side] = values
        labels = {
            key: _json_safe(handle.attrs[key])
            for key in (
                "task", "llm_description", "llm_description2", "llm_objects",
                "llm_verbs", "llm_type",
            )
            if key in handle.attrs
        }

    with av.open(str(video_path)) as container:
        if not container.streams.video:
            raise ValueError(f"MP4 没有视频流: {video_path}")
        stream = container.streams.video[0]
        if not stream.average_rate:
            raise ValueError(f"MP4 缺少 FPS: {video_path}")
        fps = float(stream.average_rate)
        reported_frames = int(stream.frames or 0)
        if reported_frames and reported_frames != frame_count:
            raise ValueError(
                f"HDF5 frames={frame_count}，MP4 reported_frames={reported_frames}"
            )
        width = int(stream.codec_context.width)
        height = int(stream.codec_context.height)
        video = {
            "path": str(video_path),
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "codec": str(stream.codec_context.name),
            "pix_fmt": str(stream.codec_context.pix_fmt or "") or None,
            "audio_streams": len(container.streams.audio),
        }

    masks = {
        side: np.isfinite(values[:, 0]) & (values[:, 0] >= confidence_threshold)
        for side, values in hand_confidences.items()
    }
    any_visible = masks["left"] | masks["right"]
    duration_s = frame_count / fps if fps > 0 else 0.0
    longest_visible_s = _longest_run(any_visible, True) / fps
    longest_absence_s = _longest_run(any_visible, False) / fps
    eligible_visible_s = _eligible_visible_frames(
        any_visible, int(math.ceil(minimum_duration_s * fps))
    ) / fps
    gates = {
        "duration_at_least_minimum": duration_s >= minimum_duration_s,
        "fps_at_least_minimum": fps >= minimum_fps,
        "resolution_at_least_720p": width >= minimum_width and height >= minimum_height,
        "continuous_hand_visibility_at_least_minimum": longest_visible_s >= minimum_duration_s,
        "hand_absence_not_over_limit": longest_absence_s <= maximum_absence_s,
        "no_audio": video["audio_streams"] == 0,
    }
    hand_summaries = {
        side: _confidence_summary(side, values, confidence_threshold)
        for side, values in hand_confidences.items()
    }
    active = [hand for hand in hand_summaries.values() if hand["valid_ratio"] >= 0.05]
    if not active:
        active = list(hand_summaries.values())
    visible_ratio = float(np.mean(any_visible)) if any_visible.size else 0.0
    annotation_score = float(np.clip(
        0.30 * visible_ratio
        + 0.30 * np.mean([hand["joint_values_confident_ratio"] or 0.0 for hand in active])
        + 0.20 * np.mean([hand["all_joints_confident_ratio"] or 0.0 for hand in active])
        + 0.10 * np.mean([hand["joint_confidence_mean"] or 0.0 for hand in active])
        + 0.10 * np.mean([hand["joint_confidence_p05"] or 0.0 for hand in active]),
        0.0,
        1.0,
    ))
    return {
        **selected,
        "source_format": "egodex_hdf5",
        "source_readonly": True,
        "labels": labels,
        "capabilities": EgoDexHDF5Adapter.capabilities().to_dict(),
        "video": video,
        "duration_s": duration_s,
        "hand_metrics": {
            "any_hand_visible_ratio": visible_ratio,
            "longest_visible_s": longest_visible_s,
            "longest_absence_s": longest_absence_s,
            "eligible_visible_s": eligible_visible_s,
            **hand_summaries,
        },
        "hard_gates": gates,
        "hard_pass": all(gates.values()),
        "annotation_score": annotation_score,
        "annotation_score_semantics": (
            "weak within-batch ranking signal; not ground-truth accuracy or acceptance proof"
        ),
        "provenance": {
            "adapter": "egodex_hdf5_fast_profile",
            "readonly": True,
            "loaded_arrays": "joint_confidences_only",
            "skipped_arrays": "camera_and_joint_transform_tensors",
        },
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
    fast_profile: bool = True,
    checkpoint_every: int = 25,
    inventory_cache: Optional[Path] = None,
    refresh_inventory: bool = False,
) -> Dict[str, Any]:
    """Build read-only EgoDex QC candidates and preserve every failure as evidence."""
    if workers <= 0:
        raise ValueError("workers 必须大于 0")
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every 必须大于 0")
    if not 0.0 <= hard_negative_quantile < clean_quantile <= 1.0:
        raise ValueError("质量分位需满足 0 <= hard-negative < clean <= 1")
    dataset = dataset.expanduser().resolve()
    output = output.expanduser().resolve()
    ensure_readonly_source_boundary(dataset, output)
    output.mkdir(parents=True, exist_ok=True)
    inventory_cache = (
        inventory_cache.expanduser().resolve()
        if inventory_cache else output / "inventory.jsonl"
    )
    ensure_readonly_source_boundary(dataset, inventory_cache)
    run_started = time.monotonic()
    selected = select_egodex_episodes(
        dataset,
        episodes_per_task=episodes_per_task,
        seed=seed,
        partitions=partitions,
        inventory_cache=inventory_cache,
        refresh_inventory=refresh_inventory,
    )
    inventory_elapsed_s = time.monotonic() - run_started
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
    started = time.monotonic()
    completed_this_run = 0
    profile_function = _profile_one_fast if fast_profile else _profile_one
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(profile_function, dataset, row, **kwargs): row
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
            completed_this_run += 1
            if completed_this_run % checkpoint_every == 0 or completed_this_run == len(pending):
                elapsed_s = max(time.monotonic() - started, 1e-6)
                rate = completed_this_run / elapsed_s
                remaining = len(pending) - completed_this_run
                write_jsonl(
                    output / "profiles.partial.jsonl",
                    sorted(profiles, key=lambda row: row["episode_id"]),
                )
                write_jsonl(
                    output / "errors.partial.jsonl",
                    sorted(errors, key=lambda row: row["episode_id"]),
                )
                write_json(output / "progress.json", {
                    "schema_version": "egoqc-egodex-progress-v1",
                    "selected": len(selected),
                    "reused": len(selected) - len(pending),
                    "pending_at_start": len(pending),
                    "completed_this_run": completed_this_run,
                    "remaining": remaining,
                    "elapsed_s": elapsed_s,
                    "episodes_per_s": rate,
                    "eta_s": remaining / rate if rate > 0 else None,
                    "fast_profile": fast_profile,
                    "inventory_cache": str(inventory_cache),
                    "inventory_elapsed_s": inventory_elapsed_s,
                    "total_elapsed_s": time.monotonic() - run_started,
                    "source_readonly": True,
                    "complete": remaining == 0,
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
        "profile_mode": "confidence_only" if fast_profile else "full_canonical",
        "inventory": {
            "cache": str(inventory_cache),
            "refresh": refresh_inventory,
            "selection_elapsed_s": inventory_elapsed_s,
        },
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
