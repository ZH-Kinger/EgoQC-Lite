from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from .provenance import code_version
from .report import write_json, write_jsonl
from .storage_safety import assert_derived_output


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.expanduser().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} 必须是 JSON object")
            yield row


def _taxonomy(row: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    value = row.get("task_taxonomy") or {}
    normalized = str(value.get("normalized_task") or "unknown")
    primitives = tuple(str(item) for item in value.get("interaction_primitives") or ["unknown"])
    return normalized, primitives


def build_training_sampling_weights(
    records: Path,
    output: Path,
    *,
    minimum_weight: float = 0.25,
    maximum_weight: float = 4.0,
) -> Dict[str, Any]:
    """Balance semantic, exact-task and raw-video-group exposure for SFT."""

    if not 0 < minimum_weight <= maximum_weight:
        raise ValueError("需要 0 < minimum_weight <= maximum_weight")
    output = assert_derived_output(output)
    rows = list(_read_jsonl(records))
    if not rows:
        raise ValueError("records 为空")
    task_counts: Counter[str] = Counter()
    primitive_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    parsed = []
    for row in rows:
        task, primitives = _taxonomy(row)
        group = str(row.get("split_group") or "unknown_group")
        task_counts[task] += 1
        group_counts[group] += 1
        primitive_counts.update(set(primitives))
        parsed.append((task, primitives, group))

    count = len(rows)
    task_average = count / max(1, len(task_counts))
    primitive_average = count / max(1, len(primitive_counts))
    group_average = count / max(1, len(group_counts))
    raw_weights: List[float] = []
    factors = []
    for task, primitives, group in parsed:
        rarest_primitive_count = min(primitive_counts[item] for item in set(primitives))
        task_factor = math.sqrt(task_average / task_counts[task])
        primitive_factor = math.sqrt(primitive_average / rarest_primitive_count)
        group_factor = math.sqrt(group_average / group_counts[group])
        raw = (task_factor * primitive_factor * group_factor) ** (1.0 / 3.0)
        raw_weights.append(raw)
        factors.append((task_factor, primitive_factor, group_factor))

    mean_raw = sum(raw_weights) / len(raw_weights)
    weighted: List[Dict[str, Any]] = []
    final_weights: List[float] = []
    for source, raw, factor in zip(rows, raw_weights, factors):
        weight = min(maximum_weight, max(minimum_weight, raw / mean_raw))
        row = dict(source)
        row["training_sample_weight"] = weight
        row["training_sampling_factors"] = {
            "exact_task": factor[0],
            "rarest_interaction_primitive": factor[1],
            "split_group": factor[2],
        }
        row["training_sampling_policy"] = "inverse_sqrt_geometric_mean_v1"
        weighted.append(row)
        final_weights.append(weight)

    artifact = output / "weighted-training-candidates.jsonl"
    write_jsonl(artifact, weighted)
    summary = {
        "schema_version": "egoqc-training-sampling-weights-v1",
        "records": len(rows),
        "unique_tasks": len(task_counts),
        "interaction_primitives": len(primitive_counts),
        "split_groups": len(group_counts),
        "minimum_weight": min(final_weights),
        "mean_weight": sum(final_weights) / len(final_weights),
        "maximum_weight": max(final_weights),
        "configured_weight_bounds": [minimum_weight, maximum_weight],
        "top_record_primitives": dict(primitive_counts.most_common()),
        "raw_source_readonly": True,
        "code_version": code_version(),
        "artifact": str(artifact),
    }
    write_json(output / "summary.json", summary)
    return summary
