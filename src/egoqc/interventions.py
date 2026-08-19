from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .cache import Cache
from .provenance import code_version, config_hash
from .report import write_json, write_jsonl
from .validator import load_episode_index, load_task_map, validate_episode


PLAN_SCHEMA_VERSION = "egoqc-intervention-plan-v1"
RUN_SCHEMA_VERSION = "egoqc-intervention-evidence-v1"
SUPPORTED_FAMILIES = {
    "timestamp_offset",
    "wrist_position_offset",
    "camera_translation_spike",
    "pose_representation_offset",
    "state_mask_dropout",
    "world_translation_scale",
    "beta_drift",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath((str(path), str(root))) == str(root)
    except ValueError:
        return False


def _prepare_output(dataset: Path, output: Path) -> Tuple[Path, Path]:
    dataset = dataset.expanduser().resolve()
    output = output.expanduser().resolve()
    if _inside(output, dataset):
        raise ValueError("intervention 输出不能位于原始 dataset 内部")
    output.mkdir(parents=True, exist_ok=True)
    return dataset, output


def _dataset_identity(dataset: Path) -> Dict[str, str]:
    info_path = dataset / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    info_sha256 = _sha256(info_path)
    metadata_digest = hashlib.sha256()
    metadata_paths = sorted(
        path for path in (dataset / "meta").rglob("*") if path.is_file()
    )
    for path in metadata_paths:
        metadata_digest.update(path.relative_to(dataset).as_posix().encode("utf-8"))
        metadata_digest.update(b"\0")
        metadata_digest.update(bytes.fromhex(_sha256(path)))
        metadata_digest.update(b"\0")
    metadata_sha256 = metadata_digest.hexdigest()
    return {
        "dataset_id": f"{dataset.name}:{metadata_sha256[:16]}",
        "info_sha256": info_sha256,
        "metadata_sha256": metadata_sha256,
    }


def _stable_rank(seed: int, episode_index: int) -> str:
    return hashlib.sha256(f"{seed}:{episode_index}".encode("utf-8")).hexdigest()


def _interval(length: int, config: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    interval = config.get("interval", {})
    minimum = max(1, int(interval.get("minimum_frames", 3)))
    if length < minimum:
        return None
    start = int(math.floor(length * float(interval.get("start_fraction", 0.25))))
    end = int(math.ceil(length * float(interval.get("end_fraction", 0.75))))
    start = min(max(0, start), max(0, length - minimum))
    end = min(length, max(start + minimum, end))
    if length >= minimum + 2:
        start = max(1, start)
        end = min(length - 1, end)
        if end - start < minimum:
            end = min(length - 1, start + minimum)
    return start, end


def _source_route(dataset: Path, row: Dict[str, Any]) -> Tuple[str, os.stat_result]:
    relative = (
        Path("data")
        / f"chunk-{int(row['data/chunk_index']):03d}"
        / f"file-{int(row['data/file_index']):03d}.parquet"
    )
    source = (dataset / relative).resolve()
    if not _inside(source, dataset) or not source.is_file():
        raise FileNotFoundError(source)
    return relative.as_posix(), source.stat()


def plan_qc_interventions(
    dataset: Path,
    intervention_config: Path,
    output: Path,
    *,
    episode_indices: Optional[Sequence[int]] = None,
    families: Optional[Sequence[str]] = None,
    maximum_episodes: int = 32,
    seed: int = 20260819,
) -> Dict[str, Any]:
    """Create deterministic lazy interventions without touching source bytes."""

    if maximum_episodes < 1:
        raise ValueError("maximum_episodes 必须 >= 1")
    dataset, output = _prepare_output(dataset, output)
    specification = json.loads(
        intervention_config.expanduser().read_text(encoding="utf-8")
    )
    configured = specification.get("families", {})
    requested = list(families or configured)
    unknown = sorted(set(requested) - set(configured))
    unsupported = sorted(set(requested) - SUPPORTED_FAMILIES)
    if unknown:
        raise ValueError(f"intervention config 未定义: {unknown}")
    if unsupported:
        raise ValueError(f"当前执行器不支持: {unsupported}")
    requested = [name for name in requested if configured[name].get("enabled", True)]
    if not requested:
        raise ValueError("没有启用的 intervention family")

    identity = _dataset_identity(dataset)
    episode_rows = load_episode_index(dataset).to_pylist()
    if episode_indices is not None:
        selected_ids = {int(value) for value in episode_indices}
        episode_rows = [
            row for row in episode_rows
            if int(row["episode_index"]) in selected_ids
        ]
        missing = selected_ids - {int(row["episode_index"]) for row in episode_rows}
        if missing:
            raise ValueError(f"episode 不存在: {sorted(missing)}")
    episode_rows.sort(
        key=lambda row: _stable_rank(seed, int(row["episode_index"]))
    )
    episode_rows = episode_rows[:maximum_episodes]

    records: List[Dict[str, Any]] = []
    skipped_short = 0
    for episode in episode_rows:
        episode_index = int(episode["episode_index"])
        length = int(episode["length"])
        affected = _interval(length, specification)
        if affected is None:
            skipped_short += 1
            continue
        start, end = affected
        source_file, stat = _source_route(dataset, episode)
        for family in requested:
            family_spec = configured[family]
            levels = family_spec.get("levels", {})
            if not levels:
                raise ValueError(f"{family} 没有 levels")
            for level, parameters in levels.items():
                identity_payload = (
                    f"{PLAN_SCHEMA_VERSION}:{identity['dataset_id']}:{episode_index}:"
                    f"{family}:{level}:{start}:{end}:{seed}"
                )
                intervention_id = hashlib.sha256(
                    identity_payload.encode("utf-8")
                ).hexdigest()[:24]
                records.append({
                    "schema_version": PLAN_SCHEMA_VERSION,
                    "intervention_id": intervention_id,
                    "dataset_id": identity["dataset_id"],
                    "dataset_info_sha256": identity["info_sha256"],
                    "dataset_metadata_sha256": identity["metadata_sha256"],
                    "episode_index": episode_index,
                    "episode_length": length,
                    "tasks": episode.get("tasks", []),
                    "dataset_from_index": episode.get("dataset_from_index"),
                    "dataset_to_index": episode.get("dataset_to_index"),
                    "source_data_file": source_file,
                    "source_signature": {
                        "size_bytes": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "headtail_sha256": Cache.fingerprint(
                            dataset / source_file, "headtail"
                        ),
                    },
                    "family": family,
                    "level": str(level),
                    "modality": family_spec.get("modality"),
                    "interval": {
                        "start_frame": start,
                        "end_frame_exclusive": end,
                    },
                    "parameters": parameters,
                    "expected_experts": list(
                        family_spec.get("expected_experts", [])
                    ),
                    "view": {
                        "kind": "lazy_virtual_intervention",
                        "materialized": False,
                        "raw_immutable": True,
                        "reversal": "discard_virtual_view_and_read_source",
                    },
                    "research_label": {
                        "synthetic": True,
                        "gold": False,
                        "may_change_acceptance": False,
                    },
                })

    manifest = output / "interventions.jsonl"
    write_jsonl(manifest, records)
    counts = Counter(record["family"] for record in records)
    summary = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "dataset_id": identity["dataset_id"],
        "dataset_path_stored": False,
        "intervention_config": str(intervention_config.expanduser().resolve()),
        "intervention_config_sha256": config_hash(specification),
        "episodes_selected": len(episode_rows),
        "episodes_skipped_too_short": skipped_short,
        "interventions": len(records),
        "counts_by_family": dict(sorted(counts.items())),
        "materialized_source_copies": 0,
        "raw_source_readonly": True,
        "synthetic_is_not_gold": True,
        "artifact": str(manifest),
        "code_version": code_version(),
    }
    write_json(output / "summary.json", summary)
    return summary


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.expanduser().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"manifest 第 {line_number} 行不是合法 JSON"
                ) from error


def _array(table: pa.Table, name: str, dtype: Any = np.float64) -> np.ndarray:
    column = table[name].combine_chunks()
    if pa.types.is_fixed_size_list(column.type):
        values = column.values.to_numpy(zero_copy_only=False)
        return np.asarray(values, dtype=dtype).reshape(
            len(column), column.type.list_size
        ).copy()
    if pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
        return np.asarray(column.to_pylist(), dtype=dtype).copy()
    return np.asarray(column.to_numpy(zero_copy_only=False), dtype=dtype).copy()


def _replace(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    field = table.schema.field(name)
    payload: Any = values.tolist()
    replacement = pa.array(payload, type=field.type)
    return table.set_column(table.schema.get_field_index(name), name, replacement)


def apply_intervention(
    table: pa.Table,
    family: str,
    parameters: Dict[str, Any],
    start: int,
    end: int,
) -> pa.Table:
    """Return an in-memory Arrow view; ``table`` and source Parquet stay unchanged."""

    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"未知 intervention family={family}")
    if not (0 <= start < end <= len(table)):
        raise ValueError(f"非法 intervention interval=[{start},{end}) length={len(table)}")

    if family == "timestamp_offset":
        values = _array(table, "timestamp")
        values[start:end] += float(parameters["offset_s"])
        return _replace(table, "timestamp", values)

    if family == "wrist_position_offset":
        state = _array(table, "observation.state")
        offset = np.asarray(parameters["offset_m"], dtype=np.float64).reshape(3)
        state[start:end, 0:3] += offset
        state[start:end, 61:64] += offset
        return _replace(table, "observation.state", state)

    if family == "camera_translation_spike":
        extrinsics = _array(table, "extrinsics_w2c").reshape(-1, 4, 4)
        offset = np.asarray(parameters["offset_m"], dtype=np.float64).reshape(3)
        extrinsics[start:end, :3, 3] += offset
        return _replace(table, "extrinsics_w2c", extrinsics.reshape(-1, 16))

    if family == "pose_representation_offset":
        state = _array(table, "observation.state")
        offset = np.deg2rad(float(parameters["offset_deg"]))
        state[start:end, 6:51:3] += offset
        state[start:end, 67:112:3] += offset
        return _replace(table, "observation.state", state)

    if family == "state_mask_dropout":
        mask = _array(table, "state_mask", bool)
        side_slots = {"left": 0, "right": 1}
        sides = list(parameters.get("sides", []))
        unknown_sides = set(sides) - set(side_slots)
        if unknown_sides:
            raise ValueError(f"未知 hand side: {sorted(unknown_sides)}")
        for side in sides:
            mask[start:end, side_slots[side]] = False
        return _replace(table, "state_mask", mask)

    if family == "world_translation_scale":
        scale = float(parameters["scale"])
        result = table
        for name in ("left_transl_world", "right_transl_world"):
            values = _array(result, name)
            values[start:end] *= scale
            result = _replace(result, name, values)
        return result

    state = _array(table, "observation.state")
    maximum_delta = float(parameters["maximum_delta"])
    drift = np.linspace(0.0, maximum_delta, end - start, dtype=np.float64)
    state[start:end, 51] += drift
    state[start:end, 112] += drift
    return _replace(table, "observation.state", state)


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, (bool, np.bool_)):
        return float(bool(value))
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    return None


def _evidence_vector(
    result: Any,
    start: int,
    end: int,
) -> Dict[str, Optional[float]]:
    evidence: Dict[str, Optional[float]] = {}
    issues = Counter(issue.code for issue in result.issues)
    for code, count in issues.items():
        evidence[f"issue:{code}"] = float(count)
    for name, raw in result.metrics.items():
        number = _finite_number(raw)
        if number is not None:
            evidence[f"metric:{name}"] = number

    events: Dict[str, List[int]] = defaultdict(list)
    for event in result.bad_frames:
        events[str(event.get("code"))].append(int(event["frame_index"]))
    for code, frames in events.items():
        evidence[f"bad_frame:{code}:count"] = float(len(frames))
        inside = sum(start <= frame < end for frame in frames)
        evidence[f"bad_frame:{code}:inside_ratio"] = float(inside / len(frames))
    return evidence


def _delta_rows(
    record: Dict[str, Any],
    baseline: Dict[str, Optional[float]],
    intervened: Dict[str, Optional[float]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    expected = set(record.get("expected_experts", []))
    rows: List[Dict[str, Any]] = []
    target_response = 0.0
    target_changed = 0
    non_target_changed = 0
    for expert in sorted(set(baseline) | set(intervened) | expected):
        before = baseline.get(expert)
        after = intervened.get(expert)
        if before is None and expert.startswith(("issue:", "bad_frame:")):
            before = 0.0
        if after is None and expert.startswith(("issue:", "bad_frame:")):
            after = 0.0
        delta = None if before is None or after is None else after - before
        changed = delta is not None and abs(delta) > 1e-12
        is_target = expert in expected
        if changed and is_target:
            target_changed += 1
            target_response += abs(float(delta)) / (1.0 + abs(float(before)))
        elif changed:
            non_target_changed += 1
        rows.append({
            "schema_version": RUN_SCHEMA_VERSION,
            "intervention_id": record["intervention_id"],
            "dataset_id": record["dataset_id"],
            "episode_index": record["episode_index"],
            "family": record["family"],
            "level": record["level"],
            "start_frame": record["interval"]["start_frame"],
            "end_frame_exclusive": record["interval"]["end_frame_exclusive"],
            "expert": expert,
            "expected_target": is_target,
            "baseline": before,
            "intervened": after,
            "delta": delta,
            "changed": changed,
        })
    return rows, {
        "target_hit": target_changed > 0,
        "target_changed_experts": target_changed,
        "target_response_l1_normalized": target_response,
        "non_target_changed_experts": non_target_changed,
    }


def _target_localization(
    result: Any,
    expected_experts: Sequence[str],
    start: int,
    end: int,
) -> Dict[str, Any]:
    target_codes = {
        value.split(":", 1)[1]
        for value in expected_experts
        if value.startswith("issue:")
    }
    frames = [
        int(event["frame_index"])
        for event in result.bad_frames
        if str(event.get("code")) in target_codes
    ]
    inside = [frame for frame in frames if start <= frame < end]
    return {
        "target_event_count": len(frames),
        "target_event_inside_count": len(inside),
        "target_event_interval_precision": (
            len(inside) / len(frames) if frames else None
        ),
        "target_interval_frame_recall": (
            len(set(inside)) / max(1, end - start)
        ),
    }


def _validate_source(dataset: Path, record: Dict[str, Any]) -> Path:
    source = (dataset / str(record["source_data_file"])).resolve()
    if not _inside(source, dataset):
        raise ValueError("manifest source_data_file 越过 dataset 根目录")
    stat = source.stat()
    expected = record.get("source_signature", {})
    if (
        stat.st_size != int(expected.get("size_bytes", -1))
        or stat.st_mtime_ns != int(expected.get("mtime_ns", -1))
        or Cache.fingerprint(source, "headtail")
        != str(expected.get("headtail_sha256", ""))
    ):
        raise RuntimeError(f"源 shard 在 plan 后发生变化，请重新规划: {source}")
    return source


def run_qc_interventions(
    dataset: Path,
    manifest: Path,
    qc_config: Path,
    output: Path,
    *,
    maximum_interventions: Optional[int] = None,
) -> Dict[str, Any]:
    """Replay lazy interventions and record expert deltas for Phase A research."""

    if maximum_interventions is not None and maximum_interventions < 1:
        raise ValueError("maximum_interventions 必须 >= 1")
    dataset, output = _prepare_output(dataset, output)
    qc = json.loads(qc_config.expanduser().read_text(encoding="utf-8"))
    identity = _dataset_identity(dataset)
    records = list(_read_jsonl(manifest))
    if maximum_interventions is not None:
        records = records[:maximum_interventions]
    if not records:
        raise ValueError("intervention manifest 为空")
    for record in records:
        if record.get("schema_version") != PLAN_SCHEMA_VERSION:
            raise ValueError("intervention manifest schema_version 不兼容")
        if record.get("dataset_id") != identity["dataset_id"]:
            raise RuntimeError("manifest 与当前 dataset identity 不一致")
        if record.get("dataset_info_sha256") != identity["info_sha256"]:
            raise RuntimeError("info.json 在 plan 后发生变化，请重新规划")
        if record.get("dataset_metadata_sha256") != identity["metadata_sha256"]:
            raise RuntimeError("dataset metadata 在 plan 后发生变化，请重新规划")
    records.sort(
        key=lambda record: (
            str(record["source_data_file"]),
            int(record["episode_index"]),
            str(record["family"]),
            str(record["level"]),
        )
    )

    info = json.loads((dataset / "meta" / "info.json").read_text(encoding="utf-8"))
    video_key = qc["video_key"]
    fps = float(
        info.get("fps")
        or info["features"][video_key]["info"]["video.fps"]
    )
    task_map = load_task_map(dataset)
    requested_columns = list(dict.fromkeys(qc["required_frame_columns"] + ["main_type"]))
    validated_sources: Dict[str, Path] = {}
    current_source_key: Optional[str] = None
    current_shard: Optional[pa.Table] = None
    baseline_cache: Dict[int, Any] = {}
    baseline_rows: List[Dict[str, Any]] = []
    sample_plan_rows: List[Dict[str, Any]] = []
    delta_records: List[Dict[str, Any]] = []
    run_records: List[Dict[str, Any]] = []

    for record in records:
        source_file = str(record["source_data_file"])
        if source_file not in validated_sources:
            validated_sources[source_file] = _validate_source(dataset, record)
        source = validated_sources[source_file]
        source_key = str(source)
        if source_key != current_source_key:
            parquet = pq.ParquetFile(source)
            available = set(parquet.schema_arrow.names)
            current_shard = parquet.read(
                columns=[name for name in requested_columns if name in available]
            )
            current_source_key = source_key
        assert current_shard is not None
        shard = current_shard
        episode_index = int(record["episode_index"])
        values = _array(shard, "episode_index", np.int64).reshape(-1)
        episode_table = shard.filter(pa.array(values == episode_index))
        length = int(record["episode_length"])
        if len(episode_table) != length:
            raise RuntimeError(
                f"episode {episode_index} length 在 plan 后变化: {len(episode_table)} != {length}"
            )
        file_name = str(record["source_data_file"])
        start = int(record["interval"]["start_frame"])
        end = int(record["interval"]["end_frame_exclusive"])

        if episode_index not in baseline_cache:
            baseline = validate_episode(
                episode_table,
                episode_index,
                length,
                fps,
                qc,
                file_name,
                filtered=True,
                expected_from_index=record.get("dataset_from_index"),
                expected_to_index=record.get("dataset_to_index"),
                task_map=task_map,
                expected_tasks=record.get("tasks", []),
            )
            baseline_cache[episode_index] = baseline
            vector = _evidence_vector(baseline, 0, length)
            baseline_rows.append({
                "schema_version": RUN_SCHEMA_VERSION,
                "dataset_id": record["dataset_id"],
                "episode_index": episode_index,
                "length": length,
                "tasks": record.get("tasks", []),
                "source_data_file": file_name,
                "tier": baseline.tier,
                "issue_codes": sorted({issue.code for issue in baseline.issues}),
                "bad_frames": list(baseline.bad_frames),
                "sample_frames": list(baseline.sample_frames),
                "evidence": vector,
            })
            sample_plan_rows.append({
                "episode_index": episode_index,
                "frame_indices": list(baseline.sample_frames),
                "selection_source": "ie_qc_original_baseline",
                "synthetic": False,
            })
        baseline = baseline_cache[episode_index]
        baseline_vector = _evidence_vector(baseline, start, end)
        intervened_table = apply_intervention(
            episode_table,
            str(record["family"]),
            dict(record.get("parameters", {})),
            start,
            end,
        )
        intervened_result = validate_episode(
            intervened_table,
            episode_index,
            length,
            fps,
            qc,
            f"virtual://{record['intervention_id']}/{file_name}",
            filtered=True,
            expected_from_index=record.get("dataset_from_index"),
            expected_to_index=record.get("dataset_to_index"),
            task_map=task_map,
            expected_tasks=record.get("tasks", []),
        )
        intervened_vector = _evidence_vector(intervened_result, start, end)
        deltas, response = _delta_rows(record, baseline_vector, intervened_vector)
        delta_records.extend(deltas)
        localization = _target_localization(
            intervened_result,
            record.get("expected_experts", []),
            start,
            end,
        )
        run_records.append({
            "schema_version": RUN_SCHEMA_VERSION,
            "intervention_id": record["intervention_id"],
            "dataset_id": record["dataset_id"],
            "episode_index": episode_index,
            "family": record["family"],
            "level": record["level"],
            "modality": record.get("modality"),
            "interval": record["interval"],
            "parameters": record.get("parameters", {}),
            "baseline_tier": baseline.tier,
            "intervened_tier": intervened_result.tier,
            "baseline_issue_codes": sorted({issue.code for issue in baseline.issues}),
            "intervened_issue_codes": sorted(
                {issue.code for issue in intervened_result.issues}
            ),
            "new_issue_codes": sorted(
                {issue.code for issue in intervened_result.issues}
                - {issue.code for issue in baseline.issues}
            ),
            **response,
            **localization,
            "research_only": True,
            "may_change_acceptance": False,
        })

    grouped: Dict[Tuple[int, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for record in run_records:
        grouped[(int(record["episode_index"]), str(record["family"]))][
            str(record["level"])
        ] = record
    monotonic_checks = []
    for (episode_index, family), levels in sorted(grouped.items()):
        if "low" not in levels or "high" not in levels:
            continue
        low = float(levels["low"]["target_response_l1_normalized"])
        high = float(levels["high"]["target_response_l1_normalized"])
        monotonic_checks.append({
            "episode_index": episode_index,
            "family": family,
            "low_response": low,
            "high_response": high,
            "non_decreasing": high + 1e-12 >= low,
        })

    by_family: Dict[str, Dict[str, Any]] = {}
    for family in sorted({str(record["family"]) for record in run_records}):
        values = [record for record in run_records if record["family"] == family]
        family_monotonic = [
            item for item in monotonic_checks if item["family"] == family
        ]
        localization_values = [
            float(record["target_event_interval_precision"])
            for record in values
            if record["target_event_interval_precision"] is not None
        ]
        by_family[family] = {
            "interventions": len(values),
            "target_hit_rate": float(np.mean([record["target_hit"] for record in values])),
            "mean_target_changed_experts": float(
                np.mean([record["target_changed_experts"] for record in values])
            ),
            "mean_target_event_interval_precision": (
                float(np.mean(localization_values)) if localization_values else None
            ),
            "monotonic_non_decreasing_rate": (
                float(np.mean([item["non_decreasing"] for item in family_monotonic]))
                if family_monotonic else None
            ),
        }

    write_jsonl(output / "baseline-evidence.jsonl", baseline_rows)
    write_jsonl(output / "sample-plan.jsonl", sample_plan_rows)
    write_jsonl(output / "intervention-runs.jsonl", run_records)
    write_jsonl(output / "evidence-deltas.jsonl", delta_records)
    write_jsonl(output / "monotonicity.jsonl", monotonic_checks)
    summary = {
        "schema_version": RUN_SCHEMA_VERSION,
        "dataset_id": identity["dataset_id"],
        "manifest": str(manifest.expanduser().resolve()),
        "episodes": len(baseline_cache),
        "interventions": len(run_records),
        "evidence_delta_rows": len(delta_records),
        "target_hit_rate": float(np.mean([record["target_hit"] for record in run_records])),
        "monotonic_non_decreasing_rate": (
            float(np.mean([item["non_decreasing"] for item in monotonic_checks]))
            if monotonic_checks else None
        ),
        "by_family": by_family,
        "source_shards_read": len(validated_sources),
        "materialized_source_copies": 0,
        "raw_source_readonly": True,
        "research_warning": (
            "Controlled-intervention response is synthetic evidence, not Gold accuracy "
            "and not authorization for automated acceptance or rejection."
        ),
        "artifacts": {
            "baselines": str(output / "baseline-evidence.jsonl"),
            "sample_plan": str(output / "sample-plan.jsonl"),
            "runs": str(output / "intervention-runs.jsonl"),
            "deltas": str(output / "evidence-deltas.jsonl"),
            "monotonicity": str(output / "monotonicity.jsonl"),
        },
        "code_version": code_version(),
    }
    write_json(output / "summary.json", summary)
    return summary
