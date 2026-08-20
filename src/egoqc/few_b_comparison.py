from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .report import write_json


SCHEMA_VERSION = "egoqc-few-b-comparison-v1"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path} contains a non-object JSONL row")
                rows.append(value)
    return rows


def _safe_div(numerator: float, denominator: float) -> Optional[float]:
    return numerator / denominator if denominator else None


def _weak_metrics(
    predictions: Sequence[Dict[str, Any]],
    teacher_targets: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    tp = fp = fn = tn = 0
    absolute_errors: List[float] = []
    valid = [row for row in predictions if row.get("structured_json_valid")]
    for row in valid:
        predicted = {
            str(finding[0]): float(finding[1])
            for finding in (row.get("parsed_response") or {}).get("f", [])
        }
        for task, target in teacher_targets.get(str(row.get("video_id")), {}).items():
            target = float(target)
            probability = predicted.get(task, 0.0)
            absolute_errors.append(abs(probability - target))
            expected = target >= 0.5
            actual = probability >= 0.5
            if expected and actual:
                tp += 1
            elif expected:
                fn += 1
            elif actual:
                fp += 1
            else:
                tn += 1
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    balanced = (
        (recall + specificity) / 2.0
        if recall is not None and specificity is not None
        else None
    )
    return {
        "scope": "valid_structured_outputs_only",
        "human_accuracy": False,
        "teacher_is_weak_label": True,
        "covered_clips": len(valid),
        "total_clips": len(predictions),
        "coverage": _safe_div(len(valid), len(predictions)),
        "mae": _safe_div(sum(absolute_errors), len(absolute_errors)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced,
    }


def _format(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def compare_few_b_benchmarks(
    benchmark_roots: Sequence[Path],
    teacher_manifest: Path,
    output: Path,
) -> Dict[str, Any]:
    if len(benchmark_roots) < 2:
        raise ValueError("at least two benchmark roots are required")
    teacher_targets: Dict[str, Dict[str, float]] = {}
    for row in _read_jsonl(teacher_manifest.expanduser().resolve()):
        targets = (row.get("distillation") or {}).get("targets") or {}
        teacher_targets[str(row.get("video_id"))] = {
            str(task): float(value) for task, value in targets.items()
        }
    systems = []
    canonical_ids = None
    canonical_protocol = None
    for root_value in benchmark_roots:
        root = root_value.expanduser().resolve()
        benchmark = json.loads((root / "benchmark.json").read_text(encoding="utf-8"))
        predictions = _read_jsonl(root / "predictions.jsonl")
        video_ids = [str(row.get("video_id")) for row in predictions]
        if canonical_ids is None:
            canonical_ids = video_ids
        elif video_ids != canonical_ids:
            raise ValueError("benchmark roots do not contain the same ordered video ids")
        protocol = benchmark.get("input_protocol") or {}
        comparable_protocol = {
            key: protocol.get(key)
            for key in (
                "frame_count",
                "maximum_edge",
                "selection_seed",
                "selection_strategy",
                "maximum_clips",
                "max_new_tokens",
                "wire_output_schema",
            )
        }
        if canonical_protocol is None:
            canonical_protocol = comparable_protocol
        elif comparable_protocol != canonical_protocol:
            raise ValueError("benchmark roots do not share the same input protocol")
        weak = _weak_metrics(predictions, teacher_targets)
        output_tokens = [int(row.get("output_tokens") or 0) for row in predictions]
        peak_values = [
            float(row["peak_inference_vram_mb"])
            for row in predictions
            if row.get("peak_inference_vram_mb") is not None
        ]
        systems.append(
            {
                "model_id": benchmark.get("model_id"),
                "benchmark_root": str(root),
                "parameter_count": benchmark.get("parameter_count"),
                "parameter_b": float(benchmark.get("parameter_count") or 0) / 1e9,
                "artifact_gb": float((benchmark.get("model_artifact") or {}).get("bytes") or 0) / 1e9,
                "model_memory_gib": float(benchmark.get("model_memory_mb") or 0) / 1024.0,
                "peak_inference_vram_gib": max(peak_values) / 1024.0 if peak_values else None,
                "structured_json_coverage": weak["coverage"],
                "median_output_tokens": statistics.median(output_tokens) if output_tokens else None,
                "p50_seconds": (benchmark.get("latency_seconds") or {}).get("total_p50"),
                "p95_seconds": (benchmark.get("latency_seconds") or {}).get("total_p95"),
                "video_hours_per_wall_hour": benchmark.get("video_hours_per_wall_hour"),
                "weak_teacher_agreement": weak,
            }
        )
    comparison = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pre_sft_capacity_and_speed_pilot",
        "formal_accuracy_measured": False,
        "accuracy_claim_authorized": False,
        "warning": "teacher agreement is a training diagnostic, not human Gold accuracy",
        "input_protocol": canonical_protocol,
        "video_ids_sha256": __import__("hashlib").sha256(
            "\n".join(canonical_ids or []).encode("utf-8")
        ).hexdigest(),
        "systems": systems,
        "interpretation_guards": [
            "invalid structured outputs count as abstention and reduce coverage",
            "weak-label metrics are computed only on valid structured outputs",
            "variable generated token counts confound end-to-end latency",
            "fixed-token throughput and human Gold accuracy remain required",
        ],
    }
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "comparison.json", comparison)
    columns = [
        "model_id",
        "parameter_b",
        "artifact_gb",
        "model_memory_gib",
        "peak_inference_vram_gib",
        "structured_json_coverage",
        "median_output_tokens",
        "p50_seconds",
        "p95_seconds",
        "video_hours_per_wall_hour",
        "weak_precision",
        "weak_recall",
        "weak_f1",
        "weak_balanced_accuracy",
    ]
    tsv_lines = ["\t".join(columns)]
    markdown_lines = [
        "# few-B pre-SFT comparison",
        "",
        "> Weak-label agreement is not human accuracy. Invalid JSON is treated as abstention.",
        "",
        "| Model | Params (B) | VRAM (GiB) | JSON coverage | P50 (s) | P95 (s) | video-h/wall-h | weak recall | weak F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system in systems:
        weak = system["weak_teacher_agreement"]
        flat = {
            **system,
            "weak_precision": weak["precision"],
            "weak_recall": weak["recall"],
            "weak_f1": weak["f1"],
            "weak_balanced_accuracy": weak["balanced_accuracy"],
        }
        tsv_lines.append("\t".join(_format(flat.get(column), 6) for column in columns))
        markdown_lines.append(
            "| " + " | ".join(
                [
                    str(system["model_id"]),
                    _format(system["parameter_b"], 3),
                    _format(system["model_memory_gib"], 2),
                    _format(system["structured_json_coverage"], 2),
                    _format(system["p50_seconds"], 3),
                    _format(system["p95_seconds"], 3),
                    _format(system["video_hours_per_wall_hour"], 2),
                    _format(weak["recall"], 3),
                    _format(weak["f1"], 3),
                ]
            ) + " |"
        )
    (output / "comparison.tsv").write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")
    (output / "comparison.md").write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "output": str(output),
        "systems": len(systems),
        "clips_per_system": len(canonical_ids or []),
        "formal_accuracy_measured": False,
        "accuracy_claim_authorized": False,
    }
    write_json(output / "summary.json", summary)
    return summary
