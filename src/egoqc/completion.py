from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .provenance import code_version, config_hash
from .validator import load_episode_index


COMPLETION_PLAN_VERSION = "egoqc-completion-v1"


def _write_json_atomic(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _fps(info: Dict[str, Any], video_key: str) -> Optional[float]:
    value = info.get("fps")
    if value is None:
        value = (
            info.get("features", {})
            .get(video_key, {})
            .get("info", {})
            .get("video.fps")
        )
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) and result > 0 else None


def _video_dimensions(info: Dict[str, Any], video_key: str) -> Optional[tuple[int, int]]:
    feature = info.get("features", {}).get(video_key, {})
    shape = feature.get("shape")
    if isinstance(shape, list) and len(shape) >= 2:
        return int(shape[1]), int(shape[0])
    details = feature.get("info", {})
    width, height = details.get("video.width"), details.get("video.height")
    if width and height:
        return int(width), int(height)
    return None


def _snapshot(dataset: Path) -> Dict[str, Any]:
    paths = sorted((dataset / "data").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"{dataset}/data 下没有 parquet")
    records = []
    for path in paths:
        stat = path.stat()
        parquet = pq.ParquetFile(path)
        records.append(
            {
                "path": path.relative_to(dataset).as_posix(),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "rows": int(parquet.metadata.num_rows),
                "columns": list(parquet.schema_arrow.names),
            }
        )
    info_path = dataset / "meta" / "info.json"
    info_digest = hashlib.sha256(info_path.read_bytes()).hexdigest()
    payload = {"info_sha256": info_digest, "files": records}
    signature = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**payload, "signature": signature}


def _field_plan(
    field: str,
    columns: set[str],
    fps: Optional[float],
    dimensions: Optional[tuple[int, int]],
    episode_offsets_available: bool,
) -> Dict[str, Any]:
    if field in columns:
        return {"field": field, "status": "original", "method": None, "requires": []}
    rules: Dict[str, tuple[str, str, Iterable[str], bool]] = {
        "timestamp": (
            "derived_nominal", "frame_index_div_fps", ("frame_index", "fps"),
            "frame_index" in columns and fps is not None,
        ),
        "index": (
            "derived_exact", "episode_dataset_offset_plus_frame_index",
            ("episode_index", "frame_index", "episode_index_metadata"),
            {"episode_index", "frame_index"} <= columns and episode_offsets_available,
        ),
        "main_type": (
            "defaulted", "unknown_main_hand", (), True,
        ),
        "state_mask": (
            "derived_exact", "left_right_kept_stack", ("left_kept", "right_kept"),
            {"left_kept", "right_kept"} <= columns,
        ),
        "left_kept": (
            "derived_exact", "state_mask_left", ("state_mask",), "state_mask" in columns,
        ),
        "right_kept": (
            "derived_exact", "state_mask_right", ("state_mask",), "state_mask" in columns,
        ),
        "intrinsics": (
            "derived_exact", "pinhole_intrinsics_from_fov_and_resolution",
            ("fov", "video_width", "video_height"),
            "fov" in columns and dimensions is not None,
        ),
        "left_seg_start": ("defaulted", "unused_segment_marker", (), True),
        "left_seg_end": ("defaulted", "unused_segment_marker", (), True),
        "right_seg_start": ("defaulted", "unused_segment_marker", (), True),
        "right_seg_end": ("defaulted", "unused_segment_marker", (), True),
    }
    rule = rules.get(field)
    if rule is None or not rule[3]:
        return {
            "field": field,
            "status": "missing",
            "method": None,
            "requires": list(rule[2]) if rule else [],
        }
    return {
        "field": field,
        "status": rule[0],
        "method": rule[1],
        "requires": list(rule[2]),
    }


def plan_public_completion(
    dataset: Path,
    config: Dict[str, Any],
    output: Optional[Path] = None,
) -> Dict[str, Any]:
    """Plan safe field completion without modifying or materializing source data."""
    dataset = dataset.expanduser().resolve()
    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    video_key = str(config.get("video_key", "observation.images.ego"))
    fps = _fps(info, video_key)
    dimensions = _video_dimensions(info, video_key)
    snapshot = _snapshot(dataset)
    try:
        episode_table = load_episode_index(dataset)
        episode_offsets_available = {
            "episode_index", "dataset_from_index"
        } <= set(episode_table.column_names)
    except (FileNotFoundError, OSError):
        episode_offsets_available = False

    required_fields = list(config.get("frame_schema", {}))
    file_plans = []
    projected_sets = []
    for record in snapshot["files"]:
        columns = set(record["columns"])
        fields = [
            _field_plan(
                field, columns, fps, dimensions, episode_offsets_available
            )
            for field in required_fields
        ]
        completions = [
            value for value in fields
            if value["status"] in {"derived_exact", "derived_nominal", "defaulted"}
        ]
        unresolved = [value for value in fields if value["status"] == "missing"]
        projected = columns | {value["field"] for value in completions}
        projected_sets.append(projected)
        file_plans.append(
            {
                **record,
                "completions": completions,
                "unresolved": unresolved,
            }
        )

    common_projected = set.intersection(*projected_sets) if projected_sets else set()
    features = info.get("features", {})
    allowed_uses = []
    blocked_uses: Dict[str, list[str]] = {}
    if video_key in features:
        allowed_uses.append("video_pretrain")
    hand_requirements = {"observation.state", "state_mask", "intrinsics"}
    missing_hand = sorted(hand_requirements - common_projected)
    if missing_hand:
        blocked_uses["hand_pose_training"] = missing_hand
    else:
        allowed_uses.append("hand_pose_training")
    world_requirements = {"extrinsics_w2c", "left_transl_world", "right_transl_world"}
    missing_world = sorted(world_requirements - common_projected)
    if missing_world:
        blocked_uses["world_motion"] = missing_world
    else:
        allowed_uses.append("world_motion")

    plan = {
        "plan_version": COMPLETION_PLAN_VERSION,
        "source_class": "public",
        "dataset": str(dataset),
        "source_signature": snapshot["signature"],
        "standard_version": config.get("standard_version"),
        "config_hash": config_hash(config),
        "code_version": code_version(),
        "video_key": video_key,
        "fps": fps,
        "video_dimensions": list(dimensions) if dimensions else None,
        "files": file_plans,
        "allowed_uses": sorted(allowed_uses),
        "blocked_uses": blocked_uses,
        "policy": {
            "raw_is_immutable": True,
            "estimated_fields_materialized": False,
            "nominal_timestamp_is_independent_clock": False,
            "defaulted_fields_are_ground_truth": False,
        },
    }
    if output is not None:
        _write_json_atomic(output.expanduser(), plan)
    return plan


def _episode_offsets(dataset: Path) -> Dict[int, int]:
    table = load_episode_index(dataset)
    return {
        int(episode): int(offset)
        for episode, offset in zip(
            table["episode_index"].to_pylist(),
            table["dataset_from_index"].to_pylist(),
        )
    }


def _fixed_list(values: np.ndarray, width: int, value_type: pa.DataType) -> pa.Array:
    return pa.array(values.tolist(), type=pa.list_(value_type, width))


def _derive_columns(
    table: pa.Table,
    completions: list[Dict[str, Any]],
    fps: Optional[float],
    dimensions: Optional[tuple[int, int]],
    offsets: Dict[int, int],
) -> Dict[str, pa.Array]:
    length = len(table)
    result: Dict[str, pa.Array] = {}
    for action in completions:
        field, method = action["field"], action["method"]
        if method == "frame_index_div_fps":
            frame = np.asarray(table["frame_index"].to_numpy(), dtype=np.float64)
            result[field] = pa.array(frame / float(fps), type=pa.float64())
        elif method == "episode_dataset_offset_plus_frame_index":
            episode = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
            frame = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
            values = np.asarray([offsets[int(value)] for value in episode], dtype=np.int64) + frame
            result[field] = pa.array(values, type=pa.int64())
        elif method == "unknown_main_hand":
            result[field] = pa.array(np.full(length, -1, dtype=np.int64))
        elif method == "left_right_kept_stack":
            left = np.asarray(table["left_kept"].to_numpy(), dtype=bool)
            right = np.asarray(table["right_kept"].to_numpy(), dtype=bool)
            result[field] = _fixed_list(np.column_stack((left, right)), 2, pa.bool_())
        elif method in {"state_mask_left", "state_mask_right"}:
            masks = np.asarray(table["state_mask"].to_pylist(), dtype=bool)
            result[field] = pa.array(masks[:, 0 if method.endswith("left") else 1])
        elif method == "pinhole_intrinsics_from_fov_and_resolution":
            if dimensions is None:
                raise ValueError("缺少视频尺寸，不能生成 intrinsics")
            width, height = dimensions
            fov = np.asarray(table["fov"].to_pylist(), dtype=np.float64)
            if (
                fov.shape != (length, 2)
                or not np.isfinite(fov).all()
                or np.any(fov <= 0)
                or np.any(fov >= np.pi)
            ):
                raise ValueError("fov 必须是有限的 [horizontal, vertical] 弧度且位于 (0, pi)")
            fx = width / (2.0 * np.tan(fov[:, 0] / 2.0))
            fy = height / (2.0 * np.tan(fov[:, 1] / 2.0))
            matrices = np.zeros((length, 3, 3), dtype=np.float64)
            matrices[:, 0, 0], matrices[:, 1, 1], matrices[:, 2, 2] = fx, fy, 1.0
            matrices[:, 0, 2], matrices[:, 1, 2] = (width - 1) / 2.0, (height - 1) / 2.0
            result[field] = _fixed_list(matrices.reshape(length, 9), 9, pa.float64())
        elif method == "unused_segment_marker":
            result[field] = pa.array(np.full(length, -1, dtype=np.int64))
        else:  # pragma: no cover - plan and materializer are version-locked
            raise ValueError(f"不支持 completion method: {method}")
    return result


def build_completion_overlay(
    dataset: Path,
    plan: Union[Path, Dict[str, Any]],
    output: Path,
) -> Dict[str, Any]:
    """Materialize only safe missing columns as sidecar Parquet files."""
    dataset = dataset.expanduser().resolve()
    value = (
        json.loads(plan.expanduser().read_text(encoding="utf-8"))
        if isinstance(plan, Path)
        else dict(plan)
    )
    if value.get("plan_version") != COMPLETION_PLAN_VERSION:
        raise ValueError("completion plan 版本不受支持")
    if Path(str(value.get("dataset"))).resolve() != dataset:
        raise ValueError("completion plan 不属于当前数据集")
    current = _snapshot(dataset)
    if current["signature"] != value.get("source_signature"):
        raise ValueError("源数据在 plan 后发生变化，请重新生成 completion plan")

    output = output.expanduser().resolve()
    fps = value.get("fps")
    raw_dimensions = value.get("video_dimensions")
    dimensions = tuple(int(item) for item in raw_dimensions) if raw_dimensions else None
    requires_offsets = any(
        action.get("method") == "episode_dataset_offset_plus_frame_index"
        for file_plan in value.get("files", [])
        for action in file_plan.get("completions", [])
    )
    offsets = _episode_offsets(dataset) if requires_offsets else {}
    plan_digest = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    written = []
    for file_plan in value.get("files", []):
        completions = list(file_plan.get("completions", []))
        if not completions:
            continue
        relative = Path(str(file_plan["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"completion plan 包含不安全路径: {relative}")
        source = (dataset / relative).resolve()
        if dataset not in source.parents:
            raise ValueError(f"completion source 越出数据集目录: {source}")
        required = sorted(
            {
                name
                for action in completions
                for name in action.get("requires", [])
                if name in file_plan["columns"]
            }
        )
        for key in ("episode_index", "frame_index"):
            if key in file_plan["columns"] and key not in required:
                required.append(key)
        table = pq.read_table(source, columns=required)
        derived = _derive_columns(table, completions, fps, dimensions, offsets)
        columns: Dict[str, pa.Array] = {
            "_source_row": pa.array(np.arange(len(table), dtype=np.int64))
        }
        for key in ("episode_index", "frame_index"):
            if key in table.column_names:
                columns[key] = table[key].combine_chunks()
        columns.update(derived)
        overlay = pa.table(columns).replace_schema_metadata(
            {
                b"egoqc.plan_sha256": plan_digest.encode("ascii"),
                b"egoqc.source_path": file_plan["path"].encode("utf-8"),
                b"egoqc.source_signature": value["source_signature"].encode("ascii"),
            }
        )
        destination = output / relative.parent / f"{relative.stem}.overlay.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        pq.write_table(overlay, temporary, compression="zstd")
        temporary.replace(destination)
        written.append(
            {
                "source": file_plan["path"],
                "overlay": destination.relative_to(output).as_posix(),
                "rows": len(overlay),
                "fields": list(derived),
                "completions": completions,
            }
        )

    manifest = {
        "overlay_version": COMPLETION_PLAN_VERSION,
        "dataset": str(dataset),
        "source_signature": value["source_signature"],
        "plan_sha256": plan_digest,
        "code_version": code_version(),
        "raw_modified": False,
        "files": written,
        "allowed_uses": value.get("allowed_uses", []),
        "blocked_uses": value.get("blocked_uses", {}),
    }
    _write_json_atomic(output / "completion-overlay.json", manifest)
    return manifest
