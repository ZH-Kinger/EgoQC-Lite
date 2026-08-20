from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set

from .report import write_json


SCHEMA_VERSION = "egoqc-paired-prompt-ablation-v1"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def _identity(row: Mapping[str, Any]) -> str:
    value = row.get("record_id") or row.get("video_id")
    if not value:
        raise ValueError("prediction row has no record_id or video_id")
    return str(value)


def _task_set(row: Mapping[str, Any], threshold: float) -> Set[str]:
    if not row.get("structured_json_valid"):
        return set()
    findings = (row.get("parsed_response") or {}).get("f") or []
    return {
        str(finding[0])
        for finding in findings
        if isinstance(finding, Sequence)
        and len(finding) >= 2
        and float(finding[1]) >= threshold
    }


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _summary(rows: Iterable[Mapping[str, Any]], threshold: float) -> Dict[str, Any]:
    values = list(rows)
    valid = [row for row in values if row.get("structured_json_valid")]
    abstentions = sum(bool((row.get("parsed_response") or {}).get("a")) for row in values)
    task_counts: Counter[str] = Counter()
    any_count = 0
    confidences: List[float] = []
    for row in values:
        tasks = _task_set(row, threshold)
        any_count += bool(tasks)
        task_counts.update(tasks)
        response = row.get("parsed_response") or {}
        if response.get("c") is not None:
            confidences.append(float(response["c"]))
    return {
        "clips": len(values),
        "structured_json_valid": len(valid),
        "structured_json_coverage": _safe_rate(len(valid), len(values)),
        "abstentions": abstentions,
        "abstention_rate": _safe_rate(abstentions, len(values)),
        "any_finding_rate": _safe_rate(any_count, len(values)),
        "mean_reported_confidence": statistics.mean(confidences) if confidences else None,
        "task_trigger_counts": dict(sorted(task_counts.items())),
        "task_trigger_rates": {
            task: _safe_rate(count, len(values))
            for task, count in sorted(task_counts.items())
        },
    }


def compare_paired_prompt_predictions(
    baseline_root: Path,
    candidate_root: Path,
    output: Path,
    *,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    baseline_root = baseline_root.expanduser().resolve()
    candidate_root = candidate_root.expanduser().resolve()
    baseline = {_identity(row): row for row in _read_jsonl(baseline_root / "predictions.jsonl")}
    candidate_rows = _read_jsonl(candidate_root / "predictions.jsonl")
    candidate_ids = [_identity(row) for row in candidate_rows]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate predictions contain duplicate ids")
    missing = [identity for identity in candidate_ids if identity not in baseline]
    if missing:
        raise ValueError(f"baseline is missing {len(missing)} candidate ids")
    baseline_rows = [baseline[identity] for identity in candidate_ids]

    added: Counter[str] = Counter()
    removed: Counter[str] = Counter()
    changed_task_set = changed_any = 0
    confidence_deltas: List[float] = []
    for before, after in zip(baseline_rows, candidate_rows):
        before_tasks = _task_set(before, threshold)
        after_tasks = _task_set(after, threshold)
        changed_task_set += before_tasks != after_tasks
        changed_any += bool(before_tasks) != bool(after_tasks)
        added.update(after_tasks - before_tasks)
        removed.update(before_tasks - after_tasks)
        before_confidence = (before.get("parsed_response") or {}).get("c")
        after_confidence = (after.get("parsed_response") or {}).get("c")
        if before_confidence is not None and after_confidence is not None:
            confidence_deltas.append(float(after_confidence) - float(before_confidence))

    selections = sorted(
        {
            str(row.get("selection_source") or "unknown")
            for row in candidate_rows
        }
    )
    comparison = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "paired_prompt_behavior_comparison_not_accuracy",
        "formal_accuracy_measured": False,
        "accuracy_claim_authorized": False,
        "warning": "These are unscored model behavior deltas on identical clips, not human-Gold accuracy.",
        "probability_threshold": threshold,
        "paired_clips": len(candidate_rows),
        "paired_ids_sha256": hashlib.sha256("\n".join(candidate_ids).encode("utf-8")).hexdigest(),
        "baseline_root": str(baseline_root),
        "candidate_root": str(candidate_root),
        "baseline": _summary(baseline_rows, threshold),
        "candidate": _summary(candidate_rows, threshold),
        "by_selection_source": {
            selection: {
                "baseline": _summary(
                    [row for row in baseline_rows if str(row.get("selection_source") or "unknown") == selection],
                    threshold,
                ),
                "candidate": _summary(
                    [row for row in candidate_rows if str(row.get("selection_source") or "unknown") == selection],
                    threshold,
                ),
            }
            for selection in selections
        },
        "paired_changes": {
            "changed_task_set_clips": changed_task_set,
            "changed_task_set_rate": _safe_rate(changed_task_set, len(candidate_rows)),
            "changed_any_finding_clips": changed_any,
            "changed_any_finding_rate": _safe_rate(changed_any, len(candidate_rows)),
            "added_task_counts": dict(sorted(added.items())),
            "removed_task_counts": dict(sorted(removed.items())),
            "mean_reported_confidence_delta": (
                statistics.mean(confidence_deltas) if confidence_deltas else None
            ),
        },
    }
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "comparison.json", comparison)
    return {
        "schema_version": SCHEMA_VERSION,
        "output": str(output),
        "paired_clips": len(candidate_rows),
        "changed_task_set_rate": comparison["paired_changes"]["changed_task_set_rate"],
        "formal_accuracy_measured": False,
        "accuracy_claim_authorized": False,
    }
