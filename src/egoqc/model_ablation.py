from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .provenance import code_version, config_hash
from .report import write_json, write_jsonl


SCHEMA_VERSION = "egoqc-model-ablation-plan-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _human_label_present(row: Dict[str, Any]) -> bool:
    decision = str(row.get("decision") or row.get("human_decision") or "").strip().lower()
    if decision in {"pass", "fail", "uncertain", "unmeasurable", "accept", "reject"}:
        return True
    review = row.get("review")
    if isinstance(review, dict):
        reviewer = str(review.get("reviewer_id") or review.get("reviewer") or "").strip()
        status = str(review.get("status") or review.get("decision") or "").strip()
        return bool(reviewer and status)
    return False


def _inspect_jsonl(path: Path) -> Dict[str, Any]:
    rows = 0
    valid_objects = 0
    human_labeled = 0
    source_groups = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
            valid_objects += 1
            human_labeled += int(_human_label_present(row))
            source_group = (
                row.get("supplier_id")
                or row.get("source_dataset")
                or row.get("dataset_id")
            )
            if source_group:
                source_groups.add(str(source_group))
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "bytes": resolved.stat().st_size,
        "rows": rows,
        "valid_objects": valid_objects,
        "human_labeled_rows": human_labeled,
        "source_group_count": len(source_groups),
    }


def _run_matrix(seeds: Iterable[int]) -> List[Dict[str, Any]]:
    systems = [
        {
            "system_id": "rules_only",
            "role": "non_learned_baseline",
            "model_id": None,
            "training": "none",
        },
        {
            "system_id": "compact_scout_20m",
            "role": "routing_baseline",
            "model_id": "mobilenet_v4_hybrid_medium_temporal",
            "training": "supervised_multilabel",
        },
        {
            "system_id": "qwen3_vl_2b_full_sft",
            "role": "few_b_small",
            "model_id": "Qwen/Qwen3-VL-2B-Instruct",
            "training": "full_parameter_sft",
        },
        {
            "system_id": "qwen3_vl_4b_full_sft",
            "role": "few_b_primary",
            "model_id": "Qwen/Qwen3-VL-4B-Instruct",
            "training": "full_parameter_sft",
        },
        {
            "system_id": "qwen3_vl_8b_full_sft",
            "role": "few_b_large_challenger",
            "model_id": "Qwen/Qwen3-VL-8B-Instruct",
            "training": "full_parameter_sft_fsdp_or_zero3",
        },
        {
            "system_id": "scout_to_qwen3_vl_2b",
            "role": "selective_cascade",
            "model_id": "compact_scout_20m -> Qwen/Qwen3-VL-2B-Instruct",
            "training": "reuse_frozen_component_checkpoints",
        },
        {
            "system_id": "scout_to_qwen3_vl_4b",
            "role": "production_candidate",
            "model_id": "compact_scout_20m -> Qwen/Qwen3-VL-4B-Instruct",
            "training": "reuse_frozen_component_checkpoints",
        },
    ]
    runs: List[Dict[str, Any]] = []
    for system in systems:
        system_seeds: Sequence[Optional[int]] = [None] if system["training"] == "none" else list(seeds)
        for seed in system_seeds:
            run_id = system["system_id"] if seed is None else f'{system["system_id"]}-seed-{seed}'
            runs.append(
                {
                    "run_id": run_id,
                    **system,
                    "seed": seed,
                    "status": "planned",
                    "result_status": "not_measured",
                    "input_protocol": "same_frozen_split_8_frames_max_edge_448_letterbox",
                    "output_schema": "strict_multilabel_temporal_json_v1",
                }
            )
    return runs


def _benchmark_matrix() -> List[Dict[str, Any]]:
    return [
        {
            "benchmark_id": "scout-cpu-int8",
            "system_id": "compact_scout_20m",
            "device": "cpu",
            "precision": "int8",
            "runtime": "onnxruntime_or_openvino",
        },
        {
            "benchmark_id": "qwen3-vl-2b-cpu-int4",
            "system_id": "qwen3_vl_2b_full_sft",
            "device": "cpu",
            "precision": "int4",
            "runtime": "selected_after_compatibility_probe",
        },
        {
            "benchmark_id": "qwen3-vl-4b-cpu-int4",
            "system_id": "qwen3_vl_4b_full_sft",
            "device": "cpu",
            "precision": "int4",
            "runtime": "selected_after_compatibility_probe",
        },
        {
            "benchmark_id": "qwen3-vl-2b-gpu-bf16",
            "system_id": "qwen3_vl_2b_full_sft",
            "device": "gpu",
            "precision": "bf16",
            "runtime": "pytorch_transformers",
        },
        {
            "benchmark_id": "qwen3-vl-4b-gpu-bf16",
            "system_id": "qwen3_vl_4b_full_sft",
            "device": "gpu",
            "precision": "bf16",
            "runtime": "pytorch_transformers",
        },
        {
            "benchmark_id": "qwen3-vl-8b-gpu-bf16",
            "system_id": "qwen3_vl_8b_full_sft",
            "device": "gpu",
            "precision": "bf16",
            "runtime": "pytorch_transformers",
        },
    ]


def plan_model_ablation(
    train_manifest: Path,
    validation_gold: Path,
    test_gold: Path,
    deployment_config: Path,
    output: Path,
    *,
    seeds: Sequence[int] = (17, 31, 47),
) -> Dict[str, Any]:
    """Freeze a reproducible few-B model comparison without running training."""

    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")
    sources = {
        "train": _inspect_jsonl(train_manifest.expanduser().resolve()),
        "validation_gold": _inspect_jsonl(validation_gold.expanduser().resolve()),
        "test_gold": _inspect_jsonl(test_gold.expanduser().resolve()),
    }
    config_path = deployment_config.expanduser().resolve()
    deployment = json.loads(config_path.read_text(encoding="utf-8"))
    if deployment.get("schema_version") != "egoqc-student-deployment-v2":
        raise ValueError("deployment config must use egoqc-student-deployment-v2")
    fingerprints = {item["sha256"] for item in sources.values()}
    split_fingerprints_distinct = len(fingerprints) == len(sources)
    minimum = int(
        deployment["accuracy_gates"]["minimum_auto_decisions_if_zero_observed_errors"]
    )
    gold_rows_sufficient = all(
        sources[name]["rows"] >= minimum
        for name in ("validation_gold", "test_gold")
    )
    all_gold_rows_explicitly_human = all(
        sources[name]["rows"] > 0
        and sources[name]["human_labeled_rows"] == sources[name]["rows"]
        for name in ("validation_gold", "test_gold")
    )
    blockers = []
    if not split_fingerprints_distinct:
        blockers.append("dataset_fingerprints_are_not_distinct")
    if not gold_rows_sufficient:
        blockers.append(f"validation_and_test_each_need_at_least_{minimum}_rows")
    if not all_gold_rows_explicitly_human:
        blockers.append("validation_or_test_contains_non_human_or_unreviewed_rows")
    blockers.append("per_task_positive_negative_gold_coverage_must_pass_separate_audit")
    accuracy_claim_authorized = False
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs = _run_matrix(seeds)
    benchmark = _benchmark_matrix()
    plan = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "planned_not_measured",
        "accuracy_claim_authorized": accuracy_claim_authorized,
        "accuracy_claim_blockers": blockers,
        "code_version": code_version(),
        "host_snapshot": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "deployment_config": {
            "path": str(config_path),
            "sha256": _sha256(config_path),
            "semantic_hash": config_hash(deployment),
        },
        "data": sources,
        "data_guards": {
            "split_fingerprints_distinct": split_fingerprints_distinct,
            "minimum_rows_per_gold_split": minimum,
            "gold_rows_sufficient": gold_rows_sufficient,
            "all_gold_rows_explicitly_human": all_gold_rows_explicitly_human,
            "teacher_and_synthetic_labels_allowed_in": ["train"],
            "teacher_and_synthetic_labels_forbidden_in": ["validation", "test"],
        },
        "fair_comparison": {
            "same_frozen_split": True,
            "same_clip_window": True,
            "same_frame_count": 8,
            "same_maximum_edge": 448,
            "same_structured_output": True,
            "threshold_selected_on_validation_only": True,
            "test_threshold_reselection_forbidden": True,
        },
        "run_count": len(runs),
        "benchmark_count": len(benchmark),
        "required_result_artifacts": [
            "resolved-config.json",
            "environment.json",
            "checkpoint-sha256.json",
            "predictions.jsonl",
            "metrics.json",
            "speed-memory.json",
            "errors/index.html",
            "errors/cases/*.mp4",
            "errors/cases/*.jpg",
        ],
    }
    write_json(output / "experiment-plan.json", plan)
    write_jsonl(output / "run-matrix.jsonl", runs)
    write_jsonl(output / "benchmark-matrix.jsonl", benchmark)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "output": str(output),
        "train_rows": sources["train"]["rows"],
        "validation_gold_rows": sources["validation_gold"]["rows"],
        "test_gold_rows": sources["test_gold"]["rows"],
        "runs": len(runs),
        "benchmarks": len(benchmark),
        "accuracy_claim_authorized": accuracy_claim_authorized,
        "accuracy_claim_blockers": blockers,
    }
    write_json(output / "summary.json", summary)
    return summary
