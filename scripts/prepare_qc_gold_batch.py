#!/usr/bin/env python3
"""Prepare a reproducible, source-safe Gold calibration batch from QC runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from egoqc.clip_selection import plan_qc_clips
from egoqc.queue_gold import build_queue_gold_review
from egoqc.report import write_json
from egoqc.teacher_queue import merge_teacher_queues


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.expanduser().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            yield row


def _rank(row: Dict[str, Any], seed: int) -> str:
    identity = str(row.get("dataset_id") or row.get("dataset_path") or "")
    return hashlib.sha256(f"{seed}:{identity}".encode("utf-8")).hexdigest()


def _bad_ratio(row: Dict[str, Any]) -> float:
    return float(
        (((row.get("summary") or {}).get("bad_frames") or {}).get("bad_frame_ratio"))
        or 0.0
    )


def _stratified_rows(
    rows: List[Dict[str, Any]], maximum: int | None, seed: int
) -> List[Dict[str, Any]]:
    """Sample across deterministic bad-frame quantile bins, not input order."""

    values = [row for row in rows if row.get("status") == "succeeded"]
    if maximum is None or maximum >= len(values):
        return sorted(values, key=lambda row: (_bad_ratio(row), _rank(row, seed)))
    if maximum <= 0:
        return []
    values.sort(key=lambda row: (_bad_ratio(row), _rank(row, seed)))
    bins: List[List[Dict[str, Any]]] = [[] for _ in range(5)]
    for index, row in enumerate(values):
        bin_index = min(4, index * 5 // len(values))
        bins[bin_index].append(row)
    for bin_rows in bins:
        bin_rows.sort(key=lambda row: _rank(row, seed))
    selected: List[Dict[str, Any]] = []
    cursor = 0
    while len(selected) < maximum and any(bins):
        bin_index = cursor % len(bins)
        cursor += 1
        if bins[bin_index]:
            selected.append(bins[bin_index].pop())
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--source-class", required=True)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--supplier-id")
    parser.add_argument("--maximum-datasets", type=int)
    parser.add_argument("--clips-per-dataset", type=int, default=3)
    parser.add_argument("--maximum-requests", type=int)
    parser.add_argument("--gold-events", type=int, default=180)
    parser.add_argument("--materialize-media", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    if args.clips_per_dataset < 1 or args.gold_events < 0 or args.workers < 1:
        raise ValueError("clips-per-dataset/gold-events/workers arguments are invalid")
    rows = _stratified_rows(
        list(_read_jsonl(args.run_results)), args.maximum_datasets, args.seed
    )
    if not rows:
        raise ValueError("run-results contains no successful datasets")

    output = args.output.expanduser().resolve()
    queue_paths: List[Path] = []
    planning_summaries: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        dataset = Path(str(row["dataset_path"]))
        quality_root = Path(str(row["output_path"]))
        dataset_id = str(row.get("dataset_id") or dataset.name)
        plan_output = output / "per-dataset" / dataset_id
        summary = plan_qc_clips(
            dataset,
            quality_root,
            plan_output,
            args.task_config,
            control_ratio=0.5,
            minimum_control_clips=1,
            maximum_clips=args.clips_per_dataset,
            seed=args.seed + index,
            source_class=args.source_class,
            source_dataset=args.source_dataset,
            supplier_id=args.supplier_id,
        )
        planning_summaries.append(summary)
        queue_paths.append(Path(summary["teacher_api_queue"]))

    merged = merge_teacher_queues(
        queue_paths,
        output / "merged",
        maximum_requests=args.maximum_requests,
        seed=args.seed,
    )
    gold = None
    if args.gold_events:
        gold = build_queue_gold_review(
            Path(merged["queue"]),
            output / "gold-review",
            maximum_events=args.gold_events,
            seed=args.seed + 1,
            materialize_media=args.materialize_media,
            workers=args.workers,
        )

    summary = {
        "schema_version": "egoqc-qc-gold-batch-v1",
        "run_results": str(args.run_results.expanduser().resolve()),
        "source_class": args.source_class,
        "source_dataset": args.source_dataset,
        "supplier_id": args.supplier_id,
        "selected_datasets": len(rows),
        "bad_frame_ratio_range": [min(map(_bad_ratio, rows)), max(map(_bad_ratio, rows))],
        "planned_clips": sum(int(value["clips"]) for value in planning_summaries),
        "merged": merged,
        "gold": gold,
        "raw_source_readonly": True,
        "external_api_called": False,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
