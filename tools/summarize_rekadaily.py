#!/usr/bin/env python3
"""Summarize paired RekaDaily count/quality reports without third-party deps."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _record(count_path: Path) -> Dict[str, Any]:
    video_id = count_path.name.removesuffix("-count.json")
    quality_path = count_path.with_name(f"{video_id}-quality.json")
    count = _load(count_path)
    quality = _load(quality_path) if quality_path.exists() else {}
    metadata = count.get("metadata") or {}
    probe = count.get("video_probe") or {}
    quality_probe = quality.get("video_probe") or {}
    issues = [
        {"stage": "count", **item} for item in count.get("issues", [])
    ] + [
        {"stage": "sample-quality", **item} for item in quality.get("issues", [])
        if item.get("code") not in {issue.get("code") for issue in count.get("issues", [])}
    ]
    return {
        "video_id": video_id,
        "project": metadata.get("project"),
        "source_access": count.get("source_access"),
        "source_uri": count.get("video_uri") or count.get("video_path"),
        "container": metadata.get("src_ext"),
        "width": metadata.get("width"),
        "height": metadata.get("height"),
        "fps": metadata.get("fps"),
        "duration_s": metadata.get("duration_s"),
        "file_size_bytes": metadata.get("file_size_bytes"),
        "metadata_frames": metadata.get("num_frames"),
        "counted_frames": probe.get("counted_frames"),
        "frame_delta": (
            probe.get("counted_frames") - metadata.get("num_frames")
            if probe.get("counted_frames") is not None
            and metadata.get("num_frames") is not None
            else None
        ),
        "jitter_mean_ms": probe.get("frame_interval_jitter_mean_ms"),
        "jitter_max_ms": probe.get("frame_interval_jitter_max_ms"),
        "non_monotonic_timestamps": probe.get("non_monotonic_timestamps"),
        "quality_sample_count": quality_probe.get("quality_sample_count"),
        "blur_min": quality_probe.get("blur_laplacian_variance_min"),
        "luma_min": quality_probe.get("luma_mean_min"),
        "luma_max": quality_probe.get("luma_mean_max"),
        "decision": count.get("screening_decision", "unscored"),
        "issues": issues,
        "unavailable_acceptance_metrics": count.get(
            "unavailable_acceptance_metrics", []
        ),
    }


def summarize(input_root: Path) -> Dict[str, Any]:
    records = [_record(path) for path in sorted(input_root.glob("*-count.json"))]
    decisions = Counter(record["decision"] for record in records)
    issue_counts = Counter(
        issue["code"] for record in records for issue in record["issues"]
    )
    unavailable = sorted({
        metric
        for record in records
        for metric in record["unavailable_acceptance_metrics"]
    })
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root.resolve()),
        "videos": len(records),
        "duration_hours": sum(float(record["duration_s"] or 0) for record in records) / 3600,
        "logical_gb": sum(int(record["file_size_bytes"] or 0) for record in records) / 1e9,
        "exact_frame_matches": sum(record["frame_delta"] == 0 for record in records),
        "decision_counts": dict(sorted(decisions.items())),
        "issue_counts": dict(issue_counts.most_common()),
        "unavailable_acceptance_metrics": unavailable,
        "records": records,
    }


def write_outputs(summary: Dict[str, Any], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    columns = [
        "video_id", "project", "container", "width", "height", "fps",
        "duration_s", "file_size_bytes", "metadata_frames", "counted_frames",
        "frame_delta", "jitter_mean_ms", "jitter_max_ms",
        "non_monotonic_timestamps", "quality_sample_count", "blur_min",
        "luma_min", "luma_max", "decision", "issue_codes", "source_uri",
    ]
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in summary["records"]:
            row = {column: record.get(column) for column in columns}
            row["issue_codes"] = ";".join(issue["code"] for issue in record["issues"])
            writer.writerow(row)
    lines = [
        "# RekaDaily pilot QC summary",
        "",
        f"- Videos: {summary['videos']}",
        f"- Duration: {summary['duration_hours']:.3f} h",
        f"- Logical size: {summary['logical_gb']:.3f} GB",
        f"- Exact frame matches: {summary['exact_frame_matches']}/{summary['videos']}",
        f"- Decisions: `{json.dumps(summary['decision_counts'], ensure_ascii=False)}`",
        "",
        "| video | project | fps | resolution | jitter max ms | decision | issues |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for record in summary["records"]:
        issues = ", ".join(issue["code"] for issue in record["issues"]) or "—"
        lines.append(
            f"| `{record['video_id']}` | {record['project']} | "
            f"{float(record['fps'] or 0):.3f} | {record['width']}×{record['height']} | "
            f"{float(record['jitter_max_ms'] or 0):.3f} | {record['decision']} | {issues} |"
        )
    lines.extend([
        "",
        "## Capability boundary",
        "",
        "This raw snapshot is video-only. The following acceptance metrics remain unmeasured: "
        + ", ".join(summary["unavailable_acceptance_metrics"])
        + ".",
        "",
    ])
    (output_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.input_root
    summary = summarize(args.input_root)
    write_outputs(summary, output)
    print(json.dumps({key: summary[key] for key in (
        "videos", "duration_hours", "logical_gb", "exact_frame_matches",
        "decision_counts", "issue_counts",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
