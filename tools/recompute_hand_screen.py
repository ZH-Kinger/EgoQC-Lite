#!/usr/bin/env python3
"""Recompute hand-screen temporal gates from cached detector JSONL outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from egoqc.hand_screen import render_hand_evidence, summarize_hand_samples
from egoqc.report import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra-confidence", type=float, default=0.7)
    parser.add_argument("--extra-nms-iou", type=float, default=0.5)
    parser.add_argument("--extra-persistence-s", type=float, default=0.6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for source_report in sorted(args.source.glob("*/hand-screen.json")):
        report = json.loads(source_report.read_text(encoding="utf-8"))
        source_samples = source_report.with_name("hand-samples.jsonl")
        samples = [
            json.loads(line) for line in source_samples.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        sample_fps = float(report["metrics"]["sample_fps"])
        report["metrics"] = summarize_hand_samples(
            samples,
            sample_fps,
            extra_hand_confidence=args.extra_confidence,
            extra_hand_nms_iou=args.extra_nms_iou,
            extra_hand_persistence_s=args.extra_persistence_s,
        )
        report["recomputed_from"] = str(source_report.resolve())
        destination = args.output / report["video_id"]
        destination.mkdir(parents=True, exist_ok=True)
        write_json(destination / "hand-screen.json", report)
        (destination / "hand-samples.jsonl").write_text(
            source_samples.read_text(encoding="utf-8"), encoding="utf-8"
        )
        render_hand_evidence(
            Path(report["dataset"]),
            report["metadata"],
            samples,
            report["metrics"],
            destination / "hand-evidence.jpg",
        )
        reports.append(report)
    source_summary_path = args.source / "hand-screen-summary.json"
    source_summary = (
        json.loads(source_summary_path.read_text(encoding="utf-8"))
        if source_summary_path.exists() else {}
    )
    decisions = ("candidate_for_mano", "review_before_mano", "screen_out_before_mano")
    summary = {
        **source_summary,
        "recomputed_from": str(args.source.resolve()),
        "videos": len(reports),
        "decision_counts": {
            decision: sum(
                report["metrics"]["provisional_decision"] == decision
                for report in reports
            )
            for decision in decisions
        },
        "effective_video_duration_s": sum(
            report["metrics"]["effective_video_duration_s"] for report in reports
        ),
        "extra_hand_rule": {
            "confidence": args.extra_confidence,
            "class_agnostic_nms_iou": args.extra_nms_iou,
            "minimum_persistence_s": args.extra_persistence_s,
        },
        "reports": [
            str(args.output / report["video_id"] / "hand-screen.json")
            for report in reports
        ],
    }
    write_json(args.output / "hand-screen-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
