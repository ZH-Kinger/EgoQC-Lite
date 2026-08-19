from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .provenance import code_version
from .report import write_json, write_jsonl


SCHEMA_VERSION = "egoqc-frozen-training-partition-v1"


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.expanduser().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} 必须是 JSON object")
            yield value


def _group(row: Mapping[str, Any], *, source: Path) -> str:
    group = str(row.get("split_group") or "").strip()
    if not group:
        identity = row.get("request_id") or row.get("event_id") or "unknown"
        raise ValueError(f"{source} 中 {identity} 缺少 split_group，不能安全划分")
    return group


def _stratum(row: Mapping[str, Any]) -> Tuple[str, str]:
    return (
        str(row.get("source_dataset") or "unknown_dataset"),
        str(row.get("selection_source") or "unknown_selection"),
    )


def _rank(group: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest()


def _partition_gold_groups(
    rows: Sequence[Dict[str, Any]],
    source: Path,
    *,
    validation_fraction: float,
    seed: int,
) -> Dict[str, str]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_group(row, source=source)].append(row)
    if len(grouped) < 2:
        raise ValueError("Gold 集至少需要两个 split_group，才能隔离 validation/test")

    total = Counter(_stratum(row) for row in rows)
    target = {key: value * validation_fraction for key, value in total.items()}
    validation_counts: Counter[Tuple[str, str]] = Counter()
    assignment: Dict[str, str] = {}

    # Put larger groups first, then use a seeded stable rank.  Each decision
    # minimizes distance to the desired per-source/per-selection validation mix.
    ordered = sorted(
        grouped,
        key=lambda group: (-len(grouped[group]), _rank(group, seed)),
    )
    for index, group in enumerate(ordered):
        group_counts = Counter(_stratum(row) for row in grouped[group])
        current_error = sum(
            abs(validation_counts[key] - target[key]) for key in total
        )
        validation_error = sum(
            abs(validation_counts[key] + group_counts[key] - target[key])
            for key in total
        )
        remaining_groups = len(ordered) - index - 1
        validation_groups = sum(value == "validation" for value in assignment.values())
        test_groups = sum(value == "test" for value in assignment.values())
        if remaining_groups == 0 and validation_groups == 0:
            split = "validation"
        elif remaining_groups == 0 and test_groups == 0:
            split = "test"
        elif validation_error < current_error:
            split = "validation"
        elif validation_error > current_error:
            split = "test"
        else:
            split = "validation" if int(_rank(group, seed)[-1], 16) % 2 == 0 else "test"
        assignment[group] = split
        if split == "validation":
            validation_counts.update(group_counts)

    if len(set(assignment.values())) != 2:
        # Defensive fallback for unusual highly imbalanced group sizes.
        moved = ordered[-1]
        assignment[moved] = "test" if assignment[moved] == "validation" else "validation"
    return assignment


def _counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "records": len(rows),
        "groups": len({str(row["split_group"]) for row in rows}),
        "source_counts": dict(Counter(str(row.get("source_dataset") or "unknown_dataset") for row in rows)),
        "selection_counts": dict(Counter(str(row.get("selection_source") or "unknown_selection") for row in rows)),
    }


def freeze_training_partition(
    teacher_queue: Path,
    gold_events: Path,
    output: Path,
    *,
    validation_fraction: float = 0.5,
    seed: int = 29,
) -> Dict[str, Any]:
    """Freeze leakage-safe teacher-train, Gold-validation and Gold-test sets.

    Every teacher request sharing a raw-video split group with any Gold event is
    quarantined.  Gold groups are then assigned wholly to validation or test.
    """

    if not 0.05 <= validation_fraction <= 0.95:
        raise ValueError("validation_fraction 必须在 [0.05, 0.95] 内")
    teacher_queue = teacher_queue.expanduser().resolve()
    gold_events = gold_events.expanduser().resolve()
    teacher_rows = list(_read_jsonl(teacher_queue))
    gold_rows = list(_read_jsonl(gold_events))
    if not teacher_rows:
        raise ValueError("teacher_queue 为空")
    if not gold_rows:
        raise ValueError("gold_events 为空")

    gold_assignments = _partition_gold_groups(
        gold_rows,
        gold_events,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    gold_groups = set(gold_assignments)

    train_rows: List[Dict[str, Any]] = []
    quarantine_rows: List[Dict[str, Any]] = []
    for source_row in teacher_rows:
        row = dict(source_row)
        group = _group(row, source=teacher_queue)
        if group in gold_groups:
            row["dataset_role"] = "gold_group_quarantine"
            row["quarantine_reason"] = "raw_video_group_reserved_for_gold_evaluation"
            quarantine_rows.append(row)
        else:
            row["dataset_role"] = "teacher_train_candidate"
            row["training_split"] = "train"
            train_rows.append(row)

    validation_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []
    all_gold_rows: List[Dict[str, Any]] = []
    for source_row in gold_rows:
        row = dict(source_row)
        split = gold_assignments[_group(row, source=gold_events)]
        row["evaluation_split"] = split
        row["dataset_role"] = f"gold_{split}"
        all_gold_rows.append(row)
        (validation_rows if split == "validation" else test_rows).append(row)

    train_groups = {_group(row, source=teacher_queue) for row in train_rows}
    validation_groups = {_group(row, source=gold_events) for row in validation_rows}
    test_groups = {_group(row, source=gold_events) for row in test_rows}
    intersections = {
        "train_validation": sorted(train_groups & validation_groups),
        "train_test": sorted(train_groups & test_groups),
        "validation_test": sorted(validation_groups & test_groups),
    }
    if any(intersections.values()):
        raise AssertionError(f"split_group 泄漏: {intersections}")

    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "teacher_train_queue": output / "teacher-train-queue.jsonl",
        "quarantined_teacher_requests": output / "quarantined-teacher-requests.jsonl",
        "gold_validation_events": output / "gold-validation-events.jsonl",
        "gold_test_events": output / "gold-test-events.jsonl",
        "gold_all_events": output / "gold-all-events.jsonl",
    }
    write_jsonl(artifacts["teacher_train_queue"], train_rows)
    write_jsonl(artifacts["quarantined_teacher_requests"], quarantine_rows)
    write_jsonl(artifacts["gold_validation_events"], validation_rows)
    write_jsonl(artifacts["gold_test_events"], test_rows)
    write_jsonl(artifacts["gold_all_events"], all_gold_rows)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "teacher_queue": str(teacher_queue),
        "gold_events": str(gold_events),
        "seed": seed,
        "validation_fraction_target": validation_fraction,
        "teacher_input": _counts(teacher_rows),
        "teacher_train": _counts(train_rows),
        "teacher_quarantined": _counts(quarantine_rows),
        "gold_validation": _counts(validation_rows),
        "gold_test": _counts(test_rows),
        "split_group_intersections": intersections,
        "leakage_check_passed": True,
        "raw_source_readonly": True,
        "code_version": code_version(),
        "artifacts": {key: str(value) for key, value in artifacts.items()},
    }
    write_json(output / "partition-summary.json", summary)
    return summary
