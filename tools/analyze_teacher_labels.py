#!/usr/bin/env python3
"""Summarize EgoQC visual-teacher weak labels without loading video data."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def analyze(queue_path: Path) -> dict[str, Any]:
    rows = _read_jsonl(queue_path)
    tiers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    route_counts: Counter[str] = Counter()
    allowed_uses: Counter[str] = Counter()
    missing_annotations: Counter[str] = Counter()
    finding_categories: Counter[str] = Counter()
    finding_severity: Counter[str] = Counter()
    task_positive_05: Counter[str] = Counter()
    task_positive_08: Counter[str] = Counter()
    conflicts: Counter[str] = Counter()
    missing_labels: list[str] = []

    for row in rows:
        output_path = Path(row["output_path"])
        if not output_path.is_file():
            missing_labels.append(row["request_id"])
            continue
        label = json.loads(output_path.read_text(encoding="utf-8"))
        overall = label.get("overall", {})
        tier = str(row.get("candidate_tier", "unknown"))
        route = str(overall.get("recommended_route", "unknown"))
        usable = bool(overall.get("training_usable", False))
        confidence = float(overall.get("confidence", 0.0))
        tiers[tier].append({"route": route, "usable": usable, "confidence": confidence})
        route_counts[route] += 1
        allowed_uses.update(map(str, overall.get("allowed_uses", [])))
        missing_annotations.update(map(str, label.get("missing_annotations", [])))

        if tier == "candidate-clean" and route != "accept":
            conflicts["candidate_clean_not_accepted"] += 1
        if tier == "hard-negative" and route == "accept":
            conflicts["hard_negative_accepted"] += 1
        if usable and route == "reject":
            conflicts["usable_but_rejected"] += 1
        if not usable and route == "accept":
            conflicts["unusable_but_accepted"] += 1

        for task, value in label.get("tasks", {}).items():
            probability = float(value.get("probability", 0.0))
            if probability >= 0.5:
                task_positive_05[str(task)] += 1
            if probability >= 0.8:
                task_positive_08[str(task)] += 1
        for finding in label.get("findings", []):
            finding_categories[str(finding.get("category", "unknown"))] += 1
            finding_severity[str(finding.get("severity", "unknown"))] += 1

    tier_summary: dict[str, Any] = {}
    for tier, items in sorted(tiers.items()):
        tier_summary[tier] = {
            "count": len(items),
            "routes": _counter(Counter(str(item["route"]) for item in items)),
            "training_usable": sum(bool(item["usable"]) for item in items),
            "training_unusable": sum(not bool(item["usable"]) for item in items),
            "mean_confidence": round(fmean(float(item["confidence"]) for item in items), 4),
        }

    labeled = len(rows) - len(missing_labels)
    return {
        "schema_version": "egoqc-teacher-label-analysis-v1",
        "queue": str(queue_path),
        "requests": len(rows),
        "labeled": labeled,
        "missing_label_count": len(missing_labels),
        "missing_request_ids": missing_labels,
        "routes": _counter(route_counts),
        "training_usable": sum(item["training_usable"] for item in tier_summary.values()),
        "training_unusable": sum(item["training_unusable"] for item in tier_summary.values()),
        "by_candidate_tier": tier_summary,
        "selection_teacher_conflicts": _counter(conflicts),
        "positive_tasks_probability_ge_0_5": _counter(task_positive_05),
        "positive_tasks_probability_ge_0_8": _counter(task_positive_08),
        "allowed_uses": _counter(allowed_uses),
        "missing_annotations": _counter(missing_annotations),
        "finding_severity": _counter(finding_severity),
        "top_finding_categories": dict(list(_counter(finding_categories).items())[:30]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = analyze(args.queue)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if not summary["missing_label_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
