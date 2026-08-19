from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .provenance import code_version
from .report import write_json, write_jsonl


SCHEMA_VERSION = "egoqc-qc-research-evaluation-v1"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.expanduser().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path} 第 {line_number} 行不是合法 JSON") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path} 第 {line_number} 行必须是对象")
            rows.append(value)
    return rows


def _unique(rows: Iterable[Dict[str, Any]], source: Path) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        video_id = str(row.get("video_id") or "")
        if not video_id:
            raise ValueError(f"{source} 存在缺少 video_id 的记录")
        if video_id in result:
            raise ValueError(f"{source} 中 video_id 重复: {video_id}")
        result[video_id] = row
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _wilson(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    probability = successes / total
    denominator = 1.0 + z * z / total
    centre = probability + z * z / (2.0 * total)
    margin = z * math.sqrt(
        probability * (1.0 - probability) / total
        + z * z / (4.0 * total * total)
    )
    return (
        max(0.0, (centre - margin) / denominator),
        min(1.0, (centre + margin) / denominator),
    )


def _confusion(pairs: Sequence[Tuple[float, int, Dict[str, Any]]], threshold: float) -> Dict[str, int]:
    result = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for probability, label, _ in pairs:
        predicted = probability >= threshold
        if predicted and label:
            result["tp"] += 1
        elif predicted:
            result["fp"] += 1
        elif label:
            result["fn"] += 1
        else:
            result["tn"] += 1
    return result


def _metrics(counts: Dict[str, int]) -> Dict[str, Any]:
    tp, fp, tn, fn = (counts[name] for name in ("tp", "fp", "tn", "fn"))
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    false_positive_rate = fp / (fp + tn) if fp + tn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    precision_ci = _wilson(tp, tp + fp)
    recall_ci = _wilson(tp, tp + fn)
    return {
        **counts,
        "samples": total,
        "predicted_positive": tp + fp,
        "positive_coverage": (tp + fp) / total if total else 0.0,
        "precision": precision,
        "precision_95_ci": list(precision_ci) if precision is not None else None,
        "recall": recall,
        "recall_95_ci": list(recall_ci) if recall is not None else None,
        "specificity": specificity,
        "false_positive_rate": false_positive_rate,
        "f1": f1,
    }


def _ranking_metrics(pairs: Sequence[Tuple[float, int, Dict[str, Any]]]) -> Dict[str, Optional[float]]:
    if not pairs:
        return {"average_precision": None, "auroc": None, "brier_score": None, "ece_15_bin": None}
    labels = [label for _, label, _ in pairs]
    positives = sum(labels)
    negatives = len(labels) - positives
    ranked = sorted(pairs, key=lambda value: value[0], reverse=True)
    average_precision = None
    if positives:
        true_so_far = 0
        precision_sum = 0.0
        for rank, (_, label, _) in enumerate(ranked, 1):
            true_so_far += label
            if label:
                precision_sum += true_so_far / rank
        average_precision = precision_sum / positives

    auroc = None
    if positives and negatives:
        # Mann-Whitney form with average ranks for tied probabilities.
        ascending = sorted((probability, label) for probability, label, _ in pairs)
        positive_rank_sum = 0.0
        index = 0
        while index < len(ascending):
            end = index + 1
            while end < len(ascending) and ascending[end][0] == ascending[index][0]:
                end += 1
            average_rank = ((index + 1) + end) / 2.0
            positive_rank_sum += average_rank * sum(label for _, label in ascending[index:end])
            index = end
        auroc = (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)

    brier = sum((probability - label) ** 2 for probability, label, _ in pairs) / len(pairs)
    ece = 0.0
    for bin_index in range(15):
        lower, upper = bin_index / 15, (bin_index + 1) / 15
        bucket = [
            (probability, label)
            for probability, label, _ in pairs
            if lower <= probability < upper or (upper == 1 and probability == 1)
        ]
        if bucket:
            confidence = sum(value[0] for value in bucket) / len(bucket)
            accuracy = sum(value[1] for value in bucket) / len(bucket)
            ece += len(bucket) / len(pairs) * abs(confidence - accuracy)
    return {
        "average_precision": average_precision,
        "auroc": auroc,
        "brier_score": brier,
        "ece_15_bin": ece,
    }


def _select_threshold(
    pairs: Sequence[Tuple[float, int, Dict[str, Any]]],
    target_precision: float,
) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
    candidates = sorted({probability for probability, _, _ in pairs}, reverse=True)
    feasible: List[Tuple[float, float, float, Dict[str, Any]]] = []
    for threshold in candidates:
        report = _metrics(_confusion(pairs, threshold))
        lower = report["precision_95_ci"][0] if report["precision_95_ci"] else 0.0
        if report["tp"] > 0 and lower >= target_precision:
            feasible.append((float(report["recall"] or 0.0), lower, -threshold, report))
    if not feasible:
        return None, None
    _, _, negative_threshold, report = max(feasible)
    return -negative_threshold, report


def _pairs(
    predictions: Dict[str, Dict[str, Any]],
    gold: Dict[str, Dict[str, Any]],
    task: str,
) -> List[Tuple[float, int, Dict[str, Any]]]:
    result = []
    for video_id, gold_row in gold.items():
        if task not in (gold_row.get("labels") or {}) or video_id not in predictions:
            continue
        raw = (predictions[video_id].get("probabilities") or {}).get(task)
        if raw is None:
            continue
        probability = float(raw)
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(f"预测概率越界: video_id={video_id}, task={task}")
        result.append((probability, int(bool(gold_row["labels"][task])), gold_row))
    return result


def _group_value(row: Dict[str, Any], fields: Sequence[str]) -> str:
    for field in fields:
        value = row.get(field)
        if value not in (None, ""):
            return f"{field}:{value}"
    return "unknown"


def _gold_contract_issues(
    rows: Dict[str, Dict[str, Any]], split: str, identity_fields: Sequence[str]
) -> List[Dict[str, Any]]:
    issues = []
    for video_id, row in rows.items():
        missing = [
            field for field in ("reviewer_id", "reviewed_at", "label_version")
            if not row.get(field)
        ]
        if missing:
            issues.append({
                "code": "gold_provenance_incomplete",
                "split": split,
                "video_id": video_id,
                "missing": missing,
            })
        if _group_value(row, identity_fields) == "unknown":
            issues.append({
                "code": "gold_identity_group_missing",
                "split": split,
                "video_id": video_id,
            })
    return issues


def _cluster_bootstrap(
    pairs: Sequence[Tuple[float, int, Dict[str, Any]]],
    threshold: float,
    group_fields: Sequence[str],
    replicates: int,
    seed: int,
) -> Dict[str, Optional[List[float]]]:
    grouped: Dict[str, List[Tuple[float, int, Dict[str, Any]]]] = defaultdict(list)
    for pair in pairs:
        grouped[_group_value(pair[2], group_fields)].append(pair)
    if len(grouped) < 2 or replicates < 1:
        return {"precision_95_ci": None, "recall_95_ci": None, "f1_95_ci": None}
    group_counts = [_confusion(values, threshold) for values in grouped.values()]
    rng = random.Random(seed)
    samples: Dict[str, List[float]] = {"precision": [], "recall": [], "f1": []}
    for _ in range(replicates):
        total = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        for _ in group_counts:
            selected = group_counts[rng.randrange(len(group_counts))]
            for name in total:
                total[name] += selected[name]
        report = _metrics(total)
        for name in samples:
            value = report[name]
            if value is not None:
                samples[name].append(float(value))
    return {
        f"{name}_95_ci": (
            [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
            if values else None
        )
        for name, values in samples.items()
    }


def _group_reports(
    pairs: Sequence[Tuple[float, int, Dict[str, Any]]],
    threshold: float,
    fields: Sequence[str],
    minimum_group_samples: int,
) -> List[Dict[str, Any]]:
    reports = []
    for field in fields:
        grouped: Dict[str, List[Tuple[float, int, Dict[str, Any]]]] = defaultdict(list)
        for pair in pairs:
            grouped[str(pair[2].get(field) or "unknown")].append(pair)
        for value, values in sorted(grouped.items()):
            report = _metrics(_confusion(values, threshold))
            reports.append({
                "field": field,
                "value": value,
                "eligible_for_worst_group": len(values) >= minimum_group_samples,
                **report,
            })
    return reports


def evaluate_qc_research_protocol(
    validation_predictions: Path,
    validation_gold: Path,
    test_predictions: Path,
    test_gold: Path,
    task_config: Path,
    output: Path,
    *,
    group_fields: Sequence[str] = ("supplier_id", "camera_id", "source_dataset"),
    bootstrap_group_fields: Sequence[str] = (
        "person_id",
        "operator_id",
        "collection_session_id",
        "supplier_id",
    ),
    bootstrap_replicates: int = 1000,
    minimum_group_samples: int = 30,
    seed: int = 20260819,
) -> Dict[str, Any]:
    """Freeze thresholds on validation and report a single untouched-test evaluation.

    This command is intentionally separate from the engineering evaluator. It rejects
    overlapping validation/test identities and records input fingerprints so a paper
    result can be reproduced exactly.
    """

    paths = [validation_predictions, validation_gold, test_predictions, test_gold, task_config]
    if bootstrap_replicates < 0 or minimum_group_samples < 1:
        raise ValueError("bootstrap_replicates 必须 >=0，minimum_group_samples 必须 >=1")
    validation_prediction_rows = _unique(_read_jsonl(validation_predictions), validation_predictions)
    validation_gold_rows = _unique(_read_jsonl(validation_gold), validation_gold)
    test_prediction_rows = _unique(_read_jsonl(test_predictions), test_predictions)
    test_gold_rows = _unique(_read_jsonl(test_gold), test_gold)
    config = json.loads(task_config.read_text(encoding="utf-8"))
    tasks = config["model_tasks"]

    overlapping_videos = sorted(set(validation_gold_rows) & set(test_gold_rows))
    validation_groups = {
        _group_value(row, bootstrap_group_fields) for row in validation_gold_rows.values()
    } - {"unknown"}
    test_groups = {
        _group_value(row, bootstrap_group_fields) for row in test_gold_rows.values()
    } - {"unknown"}
    overlapping_groups = sorted(validation_groups & test_groups)
    contract_issues = _gold_contract_issues(
        validation_gold_rows, "validation", bootstrap_group_fields
    ) + _gold_contract_issues(test_gold_rows, "test", bootstrap_group_fields)
    protocol_blockers = []
    if overlapping_videos:
        protocol_blockers.append("validation_test_video_overlap")
    if overlapping_groups:
        protocol_blockers.append("validation_test_identity_group_overlap")
    if contract_issues:
        protocol_blockers.append("gold_contract_incomplete")

    task_reports: Dict[str, Dict[str, Any]] = {}
    per_group_rows: List[Dict[str, Any]] = []
    for task_index, (task, task_policy) in enumerate(tasks.items()):
        target_precision = float(task_policy["minimum_auto_reject_precision"])
        validation_pairs = _pairs(validation_prediction_rows, validation_gold_rows, task)
        test_pairs = _pairs(test_prediction_rows, test_gold_rows, task)
        threshold, validation_operating_point = _select_threshold(
            validation_pairs, target_precision
        )
        validation_report = {
            **_ranking_metrics(validation_pairs),
            "gold_samples": len(validation_pairs),
            "gold_positives": sum(label for _, label, _ in validation_pairs),
            "gold_negatives": sum(not label for _, label, _ in validation_pairs),
            "selected_threshold": threshold,
            "operating_point": validation_operating_point,
        }
        if threshold is None:
            task_reports[task] = {
                "target_precision": target_precision,
                "validation": validation_report,
                "test": None,
                "auto_reject_authorized": False,
                "block_reasons": ["validation_precision_confidence_bound_below_target"],
            }
            continue

        test_operating_point = _metrics(_confusion(test_pairs, threshold))
        test_operating_point["cluster_bootstrap"] = _cluster_bootstrap(
            test_pairs,
            threshold,
            bootstrap_group_fields,
            bootstrap_replicates,
            seed + task_index,
        )
        groups = _group_reports(test_pairs, threshold, group_fields, minimum_group_samples)
        for group in groups:
            per_group_rows.append({"task": task, "threshold": threshold, **group})
        eligible = [group for group in groups if group["eligible_for_worst_group"]]
        precision_values = [group["precision"] for group in eligible if group["precision"] is not None]
        recall_values = [group["recall"] for group in eligible if group["recall"] is not None]
        worst_group = {
            "minimum_samples": minimum_group_samples,
            "eligible_groups": len(eligible),
            "precision": min(precision_values) if precision_values else None,
            "recall": min(recall_values) if recall_values else None,
        }
        test_report = {
            **_ranking_metrics(test_pairs),
            "gold_samples": len(test_pairs),
            "gold_positives": sum(label for _, label, _ in test_pairs),
            "gold_negatives": sum(not label for _, label, _ in test_pairs),
            "operating_point": test_operating_point,
            "worst_group": worst_group,
        }
        lower = (
            test_operating_point["precision_95_ci"][0]
            if test_operating_point["precision_95_ci"] else 0.0
        )
        block_reasons = list(protocol_blockers)
        if lower < target_precision:
            block_reasons.append("test_precision_confidence_bound_below_target")
        if not test_pairs:
            block_reasons.append("empty_test_task")
        task_reports[task] = {
            "target_precision": target_precision,
            "validation": validation_report,
            "test": test_report,
            "auto_reject_authorized": not block_reasons,
            "block_reasons": block_reasons,
        }

    enabled = [task for task, value in task_reports.items() if value["auto_reject_authorized"]]
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "qc-research-per-group.jsonl", per_group_rows)
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "threshold_selection": "validation_only_wilson_95_lower_bound",
            "test_policy": "single_locked_evaluation_no_threshold_refit",
            "target": "per_task_auto_reject_precision",
            "cluster_bootstrap_replicates": bootstrap_replicates,
            "bootstrap_group_precedence": list(bootstrap_group_fields),
            "group_fields": list(group_fields),
            "minimum_group_samples": minimum_group_samples,
            "seed": seed,
        },
        "protocol_valid": not protocol_blockers,
        "protocol_blockers": protocol_blockers,
        "overlapping_video_ids": overlapping_videos[:100],
        "overlapping_identity_groups": overlapping_groups[:100],
        "gold_contract_issue_count": len(contract_issues),
        "gold_contract_issues": contract_issues[:100],
        "tasks": task_reports,
        "enabled_tasks": enabled,
        "all_tasks_authorized": len(enabled) == len(tasks),
        "per_group_artifact": str(output / "qc-research-per-group.jsonl"),
        "input_fingerprints": {
            str(path.expanduser().resolve()): _sha256(path) for path in paths
        },
        "code_version": code_version(),
    }
    write_json(output / "qc-research-evaluation.json", report)
    return report
