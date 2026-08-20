from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

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


def _rank(seed: int, identity: str) -> str:
    return hashlib.sha256(f"{seed}:{identity}".encode("utf-8")).hexdigest()


def _primary_primitive(row: Mapping[str, Any]) -> str:
    taxonomy = row.get("task_taxonomy") or {}
    values = taxonomy.get("interaction_primitives") or ["unknown"]
    return str(values[0])


def build_visual_intervention_plan(
    records: Path,
    config: Path,
    output: Path,
    *,
    maximum_records: int = 12_000,
    seed: int = 83,
) -> Dict[str, Any]:
    """Plan balanced lazy frame interventions; never materialize or edit raw video."""

    if maximum_records < 1:
        raise ValueError("maximum_records 必须大于 0")
    output = assert_derived_output(output)
    specification = json.loads(config.expanduser().read_text(encoding="utf-8"))
    families = specification.get("families") or {}
    if not families:
        raise ValueError("visual intervention config 没有 families")
    rows = list(_read_jsonl(records))
    if not rows:
        raise ValueError("records 为空")

    buckets: Dict[tuple[str, str], deque[Dict[str, Any]]] = {}
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (_primary_primitive(row), str(row.get("source_dataset") or "unknown"))
        grouped[key].append(row)
    for key, values in grouped.items():
        buckets[key] = deque(sorted(
            values,
            key=lambda row: _rank(seed, str(row.get("request_id") or row.get("split_group"))),
        ))
    selected: List[Dict[str, Any]] = []
    keys = sorted(buckets)
    while len(selected) < min(maximum_records, len(rows)):
        progressed = False
        for key in keys:
            if buckets[key] and len(selected) < maximum_records:
                selected.append(buckets[key].popleft())
                progressed = True
        if not progressed:
            break

    family_names = sorted(families)
    primitive_donors: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        primitive_donors[_primary_primitive(row)].append(row)
    plans = []
    family_counts = Counter()
    task_counts = Counter()
    for index, row in enumerate(selected):
        family = family_names[index % len(family_names)]
        family_spec = families[family]
        request_id = str(row.get("request_id") or row.get("split_group"))
        identity = _rank(seed + 1, f"{request_id}:{family}")[:24]
        donor_request_id: Optional[str] = None
        if family == "task_text_swap":
            current = _primary_primitive(row)
            donor_primitives = sorted(value for value in primitive_donors if value != current)
            if donor_primitives:
                donor_primitive = donor_primitives[int(identity[:8], 16) % len(donor_primitives)]
                donor_rows = primitive_donors[donor_primitive]
                donor = donor_rows[int(identity[8:16], 16) % len(donor_rows)]
                donor_request_id = str(donor.get("request_id") or donor.get("split_group"))
        target_tasks = list(family_spec.get("target_tasks") or [])
        plan = {
            "schema_version": "egoqc-lazy-visual-intervention-v1",
            "intervention_id": identity,
            "source_request_id": request_id,
            "split_group": row.get("split_group"),
            "source_dataset": row.get("source_dataset"),
            "family": family,
            "parameters": family_spec.get("parameters") or {},
            "target_tasks": target_tasks,
            "donor_request_id": donor_request_id,
            "training_sample_weight": row.get("training_sample_weight", 1.0),
            "materialization": "lazy_on_cached_frames",
            "dataset_role": "synthetic_train_only",
            "synthetic": True,
            "gold": False,
            "may_authorize_accept_or_reject": False,
            "raw_source_readonly": True,
        }
        plans.append(plan)
        family_counts[family] += 1
        task_counts.update(target_tasks)

    artifact = output / "visual-interventions.jsonl"
    write_jsonl(artifact, plans)
    summary = {
        "schema_version": "egoqc-lazy-visual-intervention-plan-v1",
        "source_records": len(rows),
        "selected_source_records": len(selected),
        "interventions": len(plans),
        "source_split_groups": len({str(row.get("split_group")) for row in selected}),
        "counts_by_family": dict(family_counts),
        "target_task_counts": dict(task_counts),
        "real_only_tasks": list(specification.get("real_only_tasks") or []),
        "lazy_on_cached_frames": True,
        "materialized_images": 0,
        "synthetic_is_not_gold": True,
        "raw_source_readonly": True,
        "code_version": code_version(),
        "artifact": str(artifact),
    }
    write_json(output / "summary.json", summary)
    return summary
