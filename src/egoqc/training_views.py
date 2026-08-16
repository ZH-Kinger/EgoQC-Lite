from __future__ import annotations

import hashlib
import json
import math
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pyarrow.parquet as pq

from .provenance import code_version
from .report import write_json, write_jsonl, write_parquet


PROFILE_VERSION = "rekadaily-training-views-v1"
DATASET_NAME = "RekaAI/RekaDaily-10k-raw"
VIDEO_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if hasattr(value, "as_py"):
        return _clean(value.as_py())
    return value


def _project_from_path(path: Path, anchor: str) -> str:
    parts = path.parts
    try:
        index = parts.index(anchor)
    except ValueError:
        return ""
    return parts[index + 1] if index + 1 < len(parts) - 1 else ""


def inventory_materialized_videos(
    dataset: Path,
    cache_path: Optional[Path] = None,
) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[str, Any]]:
    """Index downloaded videos once. Tar payload bytes are never extracted or modified."""

    inventory: Dict[Tuple[str, str], Dict[str, Any]] = {}
    loose_count = 0
    tar_member_count = 0
    tar_errors: List[Dict[str, str]] = []
    cached_shards: Dict[str, Any] = {}
    if cache_path and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("schema_version") == "materialized-inventory-v1":
                cached_shards = cached.get("shards", {})
        except (OSError, json.JSONDecodeError, AttributeError):
            cached_shards = {}
    new_cache: Dict[str, Any] = {}
    cache_hits = 0
    shards_scanned = 0

    sample_root = dataset / "sample"
    if sample_root.exists():
        for path in sample_root.rglob("*"):
            if not path.is_file() or path.suffix.lower().lstrip(".") not in VIDEO_EXTENSIONS:
                continue
            project = _project_from_path(path, "sample")
            inventory.setdefault((project, path.stem), {
                "source_uri": str(path.resolve()),
                "source_access": "loose_sample",
                "source_size_bytes": path.stat().st_size,
            })
            loose_count += 1

    data_root = dataset / "data"
    tar_files = sorted(data_root.glob("*/*.tar")) if data_root.exists() else []
    for shard in tar_files:
        project = shard.parent.name
        shard_key = str(shard.resolve())
        stat = shard.stat()
        signature = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        cached_shard = cached_shards.get(shard_key, {})
        if (
            cached_shard.get("signature") == signature
            and isinstance(cached_shard.get("members"), list)
        ):
            members = cached_shard["members"]
            for item in members:
                inventory.setdefault((project, item["video_id"]), {
                    "source_uri": item["source_uri"],
                    "source_access": "webdataset_tar_member",
                    "source_size_bytes": item["source_size_bytes"],
                })
            tar_member_count += len(members)
            cache_hits += 1
            new_cache[shard_key] = cached_shard
            continue
        shard_members: List[Dict[str, Any]] = []
        try:
            with tarfile.open(shard, "r") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    member_path = PurePosixPath(member.name)
                    if member_path.suffix.lower().lstrip(".") not in VIDEO_EXTENSIONS:
                        continue
                    source_uri = f"tar://{shard.resolve()}!/{member.name}"
                    inventory.setdefault((project, member_path.stem), {
                        "source_uri": source_uri,
                        "source_access": "webdataset_tar_member",
                        "source_size_bytes": member.size,
                    })
                    shard_members.append({
                        "video_id": member_path.stem,
                        "source_uri": source_uri,
                        "source_size_bytes": member.size,
                    })
                    tar_member_count += 1
            new_cache[shard_key] = {"signature": signature, "members": shard_members}
            shards_scanned += 1
        except (tarfile.TarError, OSError) as error:
            tar_errors.append({"path": str(shard), "error": str(error)})

    if cache_path:
        write_json(cache_path, {
            "schema_version": "materialized-inventory-v1",
            "dataset": str(dataset),
            "shards": new_cache,
        })

    return inventory, {
        "loose_videos": loose_count,
        "tar_files": len(tar_files),
        "tar_video_members": tar_member_count,
        "unique_videos": len(inventory),
        "tar_cache_hits": cache_hits,
        "tar_shards_scanned": shards_scanned,
        "tar_errors": tar_errors,
    }


def _artifact(root: Optional[Path], video_id: str, filename: str) -> Optional[Dict[str, Any]]:
    if root is None:
        return None
    candidates = (
        root / video_id / filename,
        root / f"{video_id}.json",
    )
    for path in candidates:
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"_invalid": True, "_path": str(path)}
        if isinstance(value, dict):
            value["_path"] = str(path)
            return value
    return None


def _decision(report: Optional[Dict[str, Any]]) -> str:
    if not report:
        return ""
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    return str(
        report.get("decision")
        or report.get("status")
        or metrics.get("provisional_decision")
        or ""
    ).strip().lower()


def _mano_capabilities(report: Optional[Dict[str, Any]]) -> bool:
    if not report or report.get("_invalid"):
        return False
    capabilities = report.get("capabilities", {})
    required = ("wrist_pose", "mano_pose", "betas", "state_mask")
    return all(capabilities.get(name) is True for name in required)


def _alignment_approved(report: Optional[Dict[str, Any]]) -> bool:
    if not report or report.get("_invalid"):
        return False
    accepted = _decision(report) in {"accept", "accepted", "approve", "approved", "pass", "passed"}
    reviewed = report.get("human_reviewed") is True or str(report.get("review_status", "")).lower() in {
        "reviewed", "approved",
    }
    return accepted and reviewed


def _stable_split(video_id: str) -> str:
    """Stable 95/2.5/2.5 split; adding shards never reshuffles existing samples."""

    bucket = int(hashlib.sha256(video_id.encode("utf-8")).hexdigest()[:8], 16) % 10_000
    if bucket < 9_500:
        return "train"
    if bucket < 9_750:
        return "validation"
    return "test"


def _vla_contract(
    row: Dict[str, Any],
    technical_eligible: bool,
    training_ready: bool,
    hand_report: Optional[Dict[str, Any]],
    mano_stage: str,
) -> Dict[str, Any]:
    has_text = bool(row.get("activities") or row.get("category") or row.get("subcategory"))
    has_valid_hand_signal = bool(hand_report and not hand_report.get("_invalid"))
    has_mano = mano_stage == "eligible_mano_silver"
    objectives = ["video_representation", "temporal_prediction"] if technical_eligible else []
    if technical_eligible and has_text:
        objectives.append("video_text_alignment")
    if technical_eligible and has_valid_hand_signal:
        objectives.append("hand_presence_auxiliary")
    if has_mano:
        objectives.append("mano_motion_modeling")
    return {
        "candidate": technical_eligible,
        "training_ready": training_ready,
        "split": _stable_split(str(row["video_id"])),
        "split_group": str(row["video_id"]),
        "split_warning": "source_session_id_missing",
        "allowed_objectives": objectives,
        "loss_masks": {
            "video_representation": int(technical_eligible),
            "temporal_prediction": int(technical_eligible),
            "video_text_alignment": int(technical_eligible and has_text),
            "hand_presence_auxiliary": int(technical_eligible and has_valid_hand_signal),
            "mano_motion": int(has_mano),
            "robot_action": 0,
            "camera_pose": 0,
            "tactile": 0,
        },
        "target_availability": {
            "coarse_text": has_text,
            "mano": has_mano,
            "robot_action": False,
            "camera_pose": False,
            "tactile": False,
        },
        "clip_sampler": {
            "mode": "random_window",
            "window_s": 4.0,
            "minimum_visible_duration_s": 2.0,
            "decode_fps": 8.0,
        },
    }


def _parquet_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep the operational Parquet flat and stable; JSONL remains canonical."""

    result = []
    for row in rows:
        result.append({
            "video_id": row["video_id"],
            "project": row.get("project"),
            "duration_s": row.get("duration_s"),
            "fps": row.get("fps"),
            "width": row.get("width"),
            "height": row.get("height"),
            "source_uri": row.get("source_uri"),
            "source_access": row.get("source_access"),
            "technical_status": row["video_pretrain"]["technical_status"],
            "training_ready": row["video_pretrain"]["training_ready"],
            "mano_stage": row["mano_silver"]["stage"],
            "mano_silver_ready": row["mano_silver"]["training_ready"],
            "vla_split": row["vla_pretraining"]["split"],
            "vla_training_ready": row["vla_pretraining"]["training_ready"],
            "vla_objectives_json": json.dumps(row["vla_pretraining"]["allowed_objectives"], ensure_ascii=False),
            "vla_loss_masks_json": json.dumps(row["vla_pretraining"]["loss_masks"], ensure_ascii=False),
            "reason_codes_json": json.dumps(row["video_pretrain"]["reason_codes"], ensure_ascii=False),
            "warnings_json": json.dumps(row["video_pretrain"]["warnings"], ensure_ascii=False),
            "activities_json": json.dumps(row.get("activities"), ensure_ascii=False),
            "source_revision": row["provenance"]["source_revision"],
            "code_version": row["provenance"]["code_version"],
        })
    return result


def build_rekadaily_training_views(
    dataset: Path,
    output: Path,
    *,
    materialized_only: bool = False,
    hand_screen_root: Optional[Path] = None,
    mano_root: Optional[Path] = None,
    alignment_root: Optional[Path] = None,
    minimum_duration_s: float = 5.0,
    projects: Optional[List[str]] = None,
    limit: Optional[int] = None,
    license_id: Optional[str] = None,
) -> Dict[str, Any]:
    dataset = dataset.expanduser().resolve()
    output = output.expanduser().resolve()
    if minimum_duration_s < 0:
        raise ValueError("minimum_duration_s 必须 >= 0")
    if limit is not None and limit < 1:
        raise ValueError("limit 必须 >= 1")
    index_path = dataset / "metadata" / "index.parquet"
    if not index_path.is_file():
        raise FileNotFoundError(f"缺少 RekaDaily 索引: {index_path}")

    output.mkdir(parents=True, exist_ok=True)
    inventory, inventory_summary = inventory_materialized_videos(
        dataset, output / "materialized-inventory-cache.json"
    )
    source_revision = _sha256(index_path)
    table = pq.read_table(index_path)
    raw_rows = table.to_pylist()
    project_set = set(projects or [])
    records: List[Dict[str, Any]] = []
    metadata_rows_considered = 0

    for raw in raw_rows:
        row = _clean(raw)
        project = str(row.get("project") or "")
        if project_set and project not in project_set:
            continue
        metadata_rows_considered += 1
        video_id = str(row["video_id"])
        located = inventory.get((project, video_id)) or inventory.get(("", video_id))
        if materialized_only and located is None:
            continue
        if limit is not None and len(records) >= limit:
            break

        fps = float(row.get("fps") or 0.0)
        width = int(row.get("width") or 0)
        height = int(row.get("height") or 0)
        duration = float(row.get("duration_s") or 0.0)
        extension = str(row.get("src_ext") or "").lower().lstrip(".")
        reasons: List[str] = []
        warnings: List[str] = []
        if fps < 29.9:
            reasons.append("fps_below_29_9")
        if min(width, height) < 720:
            reasons.append("short_edge_below_720")
        if duration < minimum_duration_s:
            reasons.append("duration_below_minimum")
        if located is None:
            reasons.append("raw_video_not_materialized")
        if extension not in {"mp4", "avi"}:
            warnings.append("container_requires_derived_transcode")
        if not row.get("activities"):
            warnings.append("fine_activity_label_missing")
        technical_eligible = not reasons
        governance_approved = bool(license_id)
        training_ready = technical_eligible and governance_approved

        hand_report = _artifact(hand_screen_root, video_id, "hand-screen.json")
        hand_decision = _decision(hand_report)
        mano_report = _artifact(mano_root, video_id, "mano-fit.json")
        mano_decision = _decision(mano_report)
        alignment_report = _artifact(alignment_root, video_id, "alignment-qc.json")
        mano_stage = "blocked_video_gate"
        mano_reasons = list(reasons)
        if technical_eligible:
            if not hand_report:
                mano_stage = "awaiting_hand_screen"
            elif hand_report.get("_invalid"):
                mano_stage = "invalid_hand_screen_artifact"
                mano_reasons.append("invalid_hand_screen_artifact")
            elif hand_decision == "screen_out_before_mano":
                mano_stage = "screened_out_by_hand_gate"
                mano_reasons.append("hand_screen_rejected")
            elif hand_decision == "review_before_mano":
                mano_stage = "awaiting_hand_review"
            elif hand_decision != "candidate_for_mano":
                mano_stage = "invalid_hand_screen_decision"
                mano_reasons.append("invalid_hand_screen_decision")
            elif not mano_report:
                mano_stage = "awaiting_mano_fit"
            elif mano_report.get("_invalid") or mano_decision not in {"success", "succeeded", "pass", "passed"}:
                mano_stage = "mano_fit_failed"
                mano_reasons.append("mano_fit_failed")
            elif not _mano_capabilities(mano_report):
                mano_stage = "mano_output_incomplete"
                mano_reasons.append("mano_output_incomplete")
            elif not alignment_report:
                mano_stage = "awaiting_alignment_review"
            elif not _alignment_approved(alignment_report):
                mano_stage = "alignment_not_approved"
                mano_reasons.append("alignment_not_approved")
            else:
                mano_stage = "eligible_mano_silver"

        location = located or {
            "source_uri": None,
            "source_access": "not_materialized",
            "source_size_bytes": row.get("file_size_bytes"),
        }
        record = {
            "record_id": f"rekadaily:{video_id}",
            "source_class": "public_dataset",
            "source_dataset": DATASET_NAME,
            "video_id": video_id,
            "project": project,
            "category": row.get("category"),
            "subcategory": row.get("subcategory"),
            "activities": row.get("activities"),
            "duration_s": duration,
            "fps": fps,
            "width": width,
            "height": height,
            "codec": row.get("codec"),
            "src_ext": extension,
            **location,
            "video_pretrain": {
                "technical_status": "eligible" if technical_eligible else "blocked",
                "reason_codes": reasons,
                "warnings": warnings,
                "needs_transcode": "container_requires_derived_transcode" in warnings,
                "governance_status": "approved" if governance_approved else "license_review_required",
                "license_id": license_id,
                "training_ready": training_ready,
            },
            "mano_silver": {
                "stage": mano_stage,
                "reason_codes": mano_reasons,
                "hand_screen_artifact": hand_report.get("_path") if hand_report else None,
                "mano_fit_artifact": mano_report.get("_path") if mano_report else None,
                "alignment_artifact": alignment_report.get("_path") if alignment_report else None,
                "training_ready": mano_stage == "eligible_mano_silver" and governance_approved,
            },
            "provenance": {
                "profile_version": PROFILE_VERSION,
                "source_revision": source_revision,
                "code_version": code_version(),
                "raw_immutable": True,
            },
        }
        record["vla_pretraining"] = _vla_contract(
            row, technical_eligible, training_ready, hand_report, mano_stage
        )
        records.append(record)

    candidates = [row for row in records if row["video_pretrain"]["technical_status"] == "eligible"]
    video_ready = [row for row in candidates if row["video_pretrain"]["training_ready"]]
    vla_candidates = [row for row in records if row["vla_pretraining"]["candidate"]]
    vla_ready = [row for row in records if row["vla_pretraining"]["training_ready"]]
    blocked = [row for row in records if row["video_pretrain"]["technical_status"] == "blocked"]
    mano_ready = [row for row in records if row["mano_silver"]["training_ready"]]
    queues = {
        "hand-screen-queue": [row for row in records if row["mano_silver"]["stage"] == "awaiting_hand_screen"],
        "hand-review-queue": [row for row in records if row["mano_silver"]["stage"] == "awaiting_hand_review"],
        "mano-fit-queue": [row for row in records if row["mano_silver"]["stage"] == "awaiting_mano_fit"],
        "alignment-review-queue": [row for row in records if row["mano_silver"]["stage"] == "awaiting_alignment_review"],
    }

    write_jsonl(output / "all-records.jsonl", records)
    write_parquet(output / "all-records.parquet", _parquet_rows(records))
    write_jsonl(output / "video-pretrain-candidates.jsonl", candidates)
    write_jsonl(output / "video-pretrain-ready.jsonl", video_ready)
    write_jsonl(output / "vla-pretrain-candidates.jsonl", vla_candidates)
    write_jsonl(output / "vla-pretrain-ready.jsonl", vla_ready)
    write_jsonl(output / "video-blocked.jsonl", blocked)
    write_jsonl(output / "mano-silver-ready.jsonl", mano_ready)
    for name, rows in queues.items():
        write_jsonl(output / f"{name}.jsonl", rows)

    stage_counts = Counter(row["mano_silver"]["stage"] for row in records)
    reason_counts = Counter(
        reason for row in records for reason in row["video_pretrain"]["reason_codes"]
    )
    objective_counts = Counter(
        objective
        for row in vla_candidates
        for objective in row["vla_pretraining"]["allowed_objectives"]
    )
    split_counts = Counter(row["vla_pretraining"]["split"] for row in vla_candidates)
    summary = {
        "schema_version": PROFILE_VERSION,
        "dataset": str(dataset),
        "output": str(output),
        "source_dataset": DATASET_NAME,
        "source_revision": source_revision,
        "code_version": code_version(),
        "raw_immutable": True,
        "materialized_only": materialized_only,
        "inventory": inventory_summary,
        "metadata_rows_total": table.num_rows,
        "metadata_rows_considered": metadata_rows_considered,
        "records_written": len(records),
        "minimum_duration_s": minimum_duration_s,
        "license_id": license_id,
        "video_pretrain": {
            "technical_candidates": len(candidates),
            "training_ready": len(video_ready),
            "blocked": len(blocked),
            "candidate_hours": sum(row["duration_s"] for row in candidates) / 3600.0,
            "reason_counts": dict(reason_counts),
        },
        "mano_silver": {
            "training_ready": len(mano_ready),
            "stage_counts": dict(stage_counts),
            "queue_counts": {name: len(rows) for name, rows in queues.items()},
        },
        "vla_pretraining": {
            "technical_candidates": len(vla_candidates),
            "training_ready": len(vla_ready),
            "objective_counts": dict(objective_counts),
            "split_counts": dict(split_counts),
            "robot_action_supervision": 0,
            "policy": "missing targets are masked, never synthesized as ground truth",
        },
        "artifacts": {
            "canonical": "all-records.jsonl",
            "queryable": "all-records.parquet",
            "video_candidates": "video-pretrain-candidates.jsonl",
            "video_ready": "video-pretrain-ready.jsonl",
            "mano_ready": "mano-silver-ready.jsonl",
            "vla_candidates": "vla-pretrain-candidates.jsonl",
            "vla_ready": "vla-pretrain-ready.jsonl",
        },
    }
    write_json(output / "summary.json", summary)
    return summary
