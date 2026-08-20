from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .provenance import code_version
from .report import write_json, write_jsonl
from .storage_safety import assert_derived_output


SCHEMA_VERSION = "egoqc-generality-cohort-v1"


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.expanduser().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} 必须是 JSON object")
            if not str(row.get("split_group") or "").strip():
                raise ValueError(f"{path}:{line_number} 缺少 split_group")
            yield row


def _rank(row: Mapping[str, Any], seed: int) -> str:
    identity = str(row.get("request_id") or row.get("event_id") or row["split_group"])
    return hashlib.sha256(f"{seed}:{identity}".encode("utf-8")).hexdigest()


def _stratum(row: Mapping[str, Any]) -> Tuple[str, str]:
    return (
        str(row.get("source_dataset") or "unknown_dataset"),
        str(row.get("selection_source") or "unknown_selection"),
    )


def _representatives(rows: Sequence[Dict[str, Any]], seed: int) -> Tuple[List[Dict[str, Any]], int]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["split_group"])].append(row)
    representatives = [
        min(group_rows, key=lambda row: _rank(row, seed))
        for group_rows in grouped.values()
    ]
    return representatives, len(rows) - len(representatives)


def _stratified_take(
    rows: Sequence[Dict[str, Any]], count: int, seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    count = max(0, min(count, len(rows)))
    buckets: Dict[Tuple[str, str], deque[Dict[str, Any]]] = {}
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_stratum(row)].append(row)
    for key, values in grouped.items():
        buckets[key] = deque(sorted(values, key=lambda row: _rank(row, seed)))

    selected: List[Dict[str, Any]] = []
    keys = sorted(buckets)
    while len(selected) < count:
        progressed = False
        for key in keys:
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].popleft())
                progressed = True
        if not progressed:
            break
    chosen = {str(row["split_group"]) for row in selected}
    remainder = [row for row in rows if str(row["split_group"]) not in chosen]
    return selected, remainder


def _tag(rows: Sequence[Dict[str, Any]], role: str) -> List[Dict[str, Any]]:
    tagged = []
    for source in rows:
        row = dict(source)
        row["dataset_role"] = role
        row["human_gold_status"] = "candidate_not_labeled"
        row["weak_label_only"] = True
        tagged.append(row)
    return tagged


def _summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "records": len(rows),
        "split_groups": len({str(row["split_group"]) for row in rows}),
        "source_classes": dict(Counter(str(row.get("source_class") or "unknown") for row in rows)),
        "source_datasets": dict(Counter(str(row.get("source_dataset") or "unknown") for row in rows)),
        "selection_sources": dict(Counter(str(row.get("selection_source") or "unknown") for row in rows)),
    }


def _recover_source_identity(source: Mapping[str, Any]) -> Dict[str, Any]:
    """Recover dataset/task identity hidden by legacy oss-unclassified queues."""

    row = dict(source)
    if str(row.get("source_dataset")) != "oss-unclassified":
        return row
    uri = str(row.get("raw_source_uri") or row.get("source_uri") or "")
    parts = Path(uri).parts
    try:
        oss_index = parts.index("oss")
        task_family = parts[oss_index + 1]
        dataset_id = parts[oss_index + 2]
    except (ValueError, IndexError):
        return row
    if not task_family or not dataset_id:
        return row
    row.update({
        "source_dataset_original": "oss-unclassified",
        "source_dataset": f"oss:{dataset_id}",
        "task_family": task_family,
        "source_class_original": row.get("source_class"),
        "source_class": "unclassified_mounted_dataset",
        "source_origin_status": "unclassified",
        "source_identity_inferred_from_uri": True,
    })
    return row


def plan_generality_cohort(
    queues: Sequence[Path],
    protocol: Path,
    output: Path,
    *,
    external_source_classes: Sequence[str] = ("public_dataset",),
    seed: int = 41,
) -> Dict[str, Any]:
    """Build leakage-safe cohort candidates from multiple normalized queues.

    This command plans human review; it never promotes weak/model labels to Gold.
    One deterministic representative is retained per raw-video split group.
    """

    if not queues:
        raise ValueError("至少需要一个 queue")
    output = assert_derived_output(output)
    config = json.loads(protocol.expanduser().read_text(encoding="utf-8"))
    gold = config["human_gold"]
    targets = {
        "systems": int(config["systems_scale_cohort"]["minimum_unique_clips"]),
        "train": int(config["model_development"]["target_unique_train_clips"]),
        "validation": int(gold["validation"]),
        "in_domain_test": int(gold["in_domain_test"]),
        "external_test": int(gold["external_source_test"]),
    }

    input_rows: List[Dict[str, Any]] = []
    for queue in queues:
        input_rows.extend(_recover_source_identity(row) for row in _read_jsonl(queue))
    representatives, dropped = _representatives(input_rows, seed)
    external_classes = set(external_source_classes)
    external = [row for row in representatives if str(row.get("source_class")) in external_classes]
    in_domain = [row for row in representatives if str(row.get("source_class")) not in external_classes]

    external_test, external_remainder = _stratified_take(external, targets["external_test"], seed + 1)
    requested_in_domain_gold = targets["validation"] + targets["in_domain_test"]
    available_in_domain_gold = min(len(in_domain), requested_in_domain_gold)
    validation_count = (
        round(available_in_domain_gold * targets["validation"] / requested_in_domain_gold)
        if requested_in_domain_gold
        else 0
    )
    validation, remainder = _stratified_take(in_domain, validation_count, seed + 2)
    in_domain_test, train_remainder = _stratified_take(
        remainder,
        min(targets["in_domain_test"], available_in_domain_gold - len(validation)),
        seed + 3,
    )
    train_candidates, unused = _stratified_take(
        train_remainder + external_remainder, targets["train"], seed + 4
    )
    systems, _ = _stratified_take(representatives, targets["systems"], seed + 5)

    artifacts = {
        "systems": output / "systems-cohort.jsonl",
        "train": output / "train-candidates.jsonl",
        "gold_validation": output / "gold-validation-candidates.jsonl",
        "gold_in_domain_test": output / "gold-in-domain-test-candidates.jsonl",
        "gold_external_test": output / "gold-external-test-candidates.jsonl",
        "unused": output / "unused-candidates.jsonl",
    }
    write_jsonl(artifacts["systems"], _tag(systems, "systems_scale_candidate"))
    write_jsonl(artifacts["train"], _tag(train_candidates, "teacher_train_candidate"))
    write_jsonl(artifacts["gold_validation"], _tag(validation, "gold_validation_candidate"))
    write_jsonl(artifacts["gold_in_domain_test"], _tag(in_domain_test, "gold_in_domain_test_candidate"))
    write_jsonl(artifacts["gold_external_test"], _tag(external_test, "gold_external_test_candidate"))
    write_jsonl(artifacts["unused"], _tag(unused, "unassigned_candidate"))

    assigned = {
        "train": train_candidates,
        "validation": validation,
        "in_domain_test": in_domain_test,
        "external_test": external_test,
    }
    group_sets = {name: {str(row["split_group"]) for row in rows} for name, rows in assigned.items()}
    intersections = {
        f"{left}_{right}": sorted(group_sets[left] & group_sets[right])
        for index, left in enumerate(group_sets)
        for right in list(group_sets)[index + 1 :]
    }
    gaps = {
        name: max(0, targets[name] - len(assigned_rows if name != "systems" else systems))
        for name, assigned_rows in {**assigned, "systems": systems}.items()
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "protocol": str(protocol.expanduser().resolve()),
        "input_queues": [str(path.expanduser().resolve()) for path in queues],
        "input_rows": len(input_rows),
        "unique_split_groups": len(representatives),
        "dropped_same_group_clips": dropped,
        "external_source_classes": sorted(external_classes),
        "targets": targets,
        "coverage_gaps": gaps,
        "systems": _summary(systems),
        "train": _summary(train_candidates),
        "gold_validation_candidates": _summary(validation),
        "gold_in_domain_test_candidates": _summary(in_domain_test),
        "gold_external_test_candidates": _summary(external_test),
        "split_group_intersections": intersections,
        "leakage_check_passed": not any(intersections.values()),
        "human_gold_created": 0,
        "candidate_labels_are_not_gold": True,
        "raw_source_readonly": True,
        "code_version": code_version(),
        "artifacts": {name: str(path) for name, path in artifacts.items()},
    }
    write_json(output / "summary.json", summary)
    return summary
