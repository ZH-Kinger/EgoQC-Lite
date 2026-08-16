from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .provenance import config_hash
from .report import write_jsonl, write_parquet
from .types import EpisodeResult, Issue


OPERATIONAL_CODES = {
    "data_read_failed",
    "episode_validation_failed",
    "video_open_failed",
    "video_probe_unavailable",
    "metadata_read_failed",
}

FORMAT_CODES = {
    "missing_directory",
    "missing_columns",
    "frame_dtype_mismatch",
    "frame_shape_mismatch",
    "missing_data_file",
    "missing_data_route",
    "missing_video_file",
    "missing_video_route",
    "feature_not_declared",
    "feature_dtype_mismatch",
    "feature_shape_mismatch",
    "missing_metadata_column",
    "missing_metadata_file",
    "task_index_unknown",
    "task_text_invalid",
    "episode_length_mismatch",
    "frame_index_not_contiguous",
    "global_index_not_contiguous",
    "global_index_range_mismatch",
    "segment_marker_invalid",
    "video_interval_mismatch",
    "video_frame_count_mismatch",
    "video_intervals_overlap",
    "aggregated_video_zero_offsets",
    "chunk_index_mismatch",
    "video_width_mismatch",
    "video_height_mismatch",
    "video_fps_mismatch",
    "video_codec_mismatch",
    "video_pix_fmt_mismatch",
    "unexpected_audio_stream",
    "parquet_file_too_large",
    "video_file_too_large",
}

NUMERIC_CODES = {
    "timestamp_mismatch",
    "non_finite_values",
    "invalid_rotation_matrix",
    "kept_mask_mismatch",
    "invalid_main_type",
    "world_camera_position_mismatch",
    "world_camera_rotation_mismatch",
    "pose_representation_mismatch",
    "beta_drift",
}

MOTION_CODES = {
    "position_jitter",
    "wrist_rotation_jitter",
    "joint_rotation_jitter",
    "temporal_spike",
    "mask_flicker",
    "pose_freeze",
    "camera_position_jitter",
    "camera_rotation_jitter",
    "low_valid_ratio",
    "hand_out_of_view_too_long",
    "insufficient_continuous_hand_visibility",
}

FIT_CODES = {
    "mano_projection_mismatch",
    "keypoint_reprojection_error",
    "mesh_mask_iou_low",
    "hand_visibility_mismatch",
}


def acceptance_for(
    issues: Sequence[Issue], config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    codes = {issue.code for issue in issues}
    error_codes = {issue.code for issue in issues if issue.severity == "error"}
    format_pass = not bool(codes & FORMAT_CODES)
    numeric_pass = not bool(error_codes & NUMERIC_CODES)
    motion_pass = not bool(codes & MOTION_CODES)
    fit_issues = codes & FIT_CODES
    fit_pass = None if not fit_issues else not bool(error_codes & FIT_CODES)
    operational_pass = not bool(codes & OPERATIONAL_CODES)

    if not operational_pass:
        decision = "retry"
    elif not format_pass:
        decision = "quarantine"
    elif not numeric_pass or fit_pass is False:
        decision = "reject"
    elif not motion_pass:
        decision = str((config or {}).get("acceptance", {}).get("motion_failure_decision", "review"))
        if decision not in {"review", "rework", "reject"}:
            raise ValueError(f"unsupported motion_failure_decision: {decision}")
    elif any(issue.severity == "warning" for issue in issues):
        decision = "review"
    else:
        decision = "accept"
    return {
        "format_pass": format_pass,
        "numeric_pass": numeric_pass,
        "motion_pass": motion_pass,
        "fit_pass": fit_pass,
        "operational_pass": operational_pass,
        "final_pass": decision == "accept",
        "decision": decision,
    }


def episode_manifest_row(
    dataset: Path,
    result: EpisodeResult,
    config: Dict[str, Any],
    dataset_issues: Sequence[Issue] = (),
) -> Dict[str, Any]:
    combined = list(result.issues) + list(dataset_issues)
    issues = [issue.to_dict() for issue in combined]
    files = sorted({issue.file for issue in combined if issue.file})
    return {
        "dataset": str(dataset),
        "episode_index": result.episode_index,
        "length": result.length,
        "tier": result.tier,
        **acceptance_for(combined, config),
        "issue_codes": sorted({issue["code"] for issue in issues}),
        "files": files,
        "sample_frames": result.sample_frames,
        "standard_version": config["standard_version"],
        "config_hash": config_hash(config),
    }


def write_decision_manifests(
    output: Path,
    dataset: Path,
    results: Sequence[EpisodeResult],
    config: Dict[str, Any],
    dataset_issues: Sequence[Issue] = (),
) -> Dict[str, Any]:
    root = output / "decisions"
    rows = [episode_manifest_row(dataset, result, config, dataset_issues) for result in results]
    by_decision: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_decision[row["decision"]].append(row)

    write_parquet(root / "episode_decisions.parquet", rows)
    write_jsonl(root / "episode_decisions.jsonl", rows)
    for decision in ("quarantine", "reject", "rework", "review", "retry"):
        write_jsonl(root / f"{decision}_manifest.jsonl", by_decision[decision])

    rejected = by_decision["quarantine"] + by_decision["reject"] + by_decision["rework"]
    write_parquet(root / "rejected_episodes.parquet", rejected)

    file_rows: List[Dict[str, Any]] = []
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if row["decision"] == "accept":
            continue
        for file_name in row["files"]:
            current = grouped.setdefault(
                file_name,
                {
                    "dataset": str(dataset),
                    "file": file_name,
                    "kind": "video" if file_name.endswith(".mp4") else "parquet",
                    "decisions": set(),
                    "episode_indices": [],
                    "issue_codes": set(),
                },
            )
            current["decisions"].add(row["decision"])
            current["episode_indices"].append(row["episode_index"])
            current["issue_codes"].update(row["issue_codes"])
    for value in grouped.values():
        file_rows.append(
            {
                **value,
                "decisions": sorted(value["decisions"]),
                "episode_indices": sorted(set(value["episode_indices"])),
                "issue_codes": sorted(value["issue_codes"]),
            }
        )
    write_parquet(root / "rejected_files.parquet", file_rows)
    write_jsonl(root / "rejected_files.jsonl", file_rows)
    write_jsonl(
        root / "retry_files.jsonl",
        [row for row in file_rows if "retry" in row["decisions"]],
    )
    return {
        "counts": {key: len(value) for key, value in sorted(by_decision.items())},
        "episode_decisions": str(root / "episode_decisions.parquet"),
        "rejected_files": str(root / "rejected_files.parquet"),
        "root": str(root),
    }


def create_retry_plan(quality_roots: Sequence[Path], output: Path) -> Dict[str, Any]:
    tasks: Dict[str, Dict[str, Any]] = {}
    for root in quality_roots:
        source = root / "decisions" / "retry_files.jsonl"
        if not source.exists():
            continue
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            identity = f"{row.get('dataset')}\0{row.get('file')}"
            task_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
            tasks[task_id] = {
                "task_id": task_id,
                "dataset": row.get("dataset"),
                "file": row.get("file"),
                "kind": row.get("kind"),
                "episode_indices": row.get("episode_indices", []),
                "issue_codes": row.get("issue_codes", []),
                "action": "rerun_dataset_with_artifact_cache",
                "source_quality_root": str(root),
            }
    values = [tasks[key] for key in sorted(tasks)]
    write_jsonl(output, values)
    return {"output": str(output.resolve()), "task_count": len(values)}
