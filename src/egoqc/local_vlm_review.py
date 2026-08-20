from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from .provenance import code_version
from .report import write_json, write_jsonl


SCHEMA_VERSION = "egoqc-local-vlm-review-queue-v1"


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.expanduser().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            yield row


def _identity(row: Mapping[str, Any]) -> str:
    return str(row.get("request_id") or row.get("video_id") or row.get("record_id") or "")


def prepare_local_vlm_review_queue(
    queue: Path,
    benchmark_root: Path,
    output: Path,
    *,
    probability_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Attach unscored local VLM suggestions to a human-review queue.

    The exported teacher-shaped labels are presentation adapters only. They
    cannot accept/reject data and must never be treated as Gold or SFT labels.
    """

    if not 0.0 <= probability_threshold <= 1.0:
        raise ValueError("probability_threshold must be between 0 and 1")
    queue = queue.expanduser().resolve()
    benchmark_root = benchmark_root.expanduser().resolve()
    benchmark = json.loads((benchmark_root / "benchmark.json").read_text(encoding="utf-8"))
    predictions_path = benchmark_root / "predictions.jsonl"
    predictions = {_identity(row): row for row in _read_jsonl(predictions_path)}
    predictions_sha256 = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
    output = output.expanduser().resolve()
    labels_root = output / "machine-suggestions"
    labels_root.mkdir(parents=True, exist_ok=True)
    prompt_version = (benchmark.get("input_protocol") or {}).get("prompt_version")
    task_order = list((benchmark.get("input_protocol") or {}).get("task_order") or [])
    rows = []
    missing = 0
    invalid = 0
    source_counts: Counter[str] = Counter()
    suggested_counts: Counter[str] = Counter()
    for request in _read_jsonl(queue):
        identity = _identity(request)
        prediction = predictions.get(identity)
        if prediction is None:
            missing += 1
            continue
        if not prediction.get("structured_json_valid"):
            invalid += 1
            continue
        parsed = prediction.get("parsed_response") or {}
        confidence = float(parsed.get("c") or 0.0)
        duration = max(
            0.0,
            float(request.get("clip_end_s") or 0.0) - float(request.get("clip_start_s") or 0.0),
        )
        task_scores = {
            task: {"probability": 0.0, "confidence": confidence}
            for task in task_order
        }
        findings = []
        predicted_tasks = []
        for finding in parsed.get("f") or []:
            if not isinstance(finding, list) or len(finding) != 6:
                continue
            task = str(finding[0])
            probability = float(finding[1])
            if task not in task_scores:
                continue
            task_scores[task] = {"probability": probability, "confidence": confidence}
            if probability >= probability_threshold:
                predicted_tasks.append(task)
                suggested_counts[task] += 1
            findings.append(
                {
                    "category": task,
                    "probability": probability,
                    "severity": int(finding[2]),
                    "start_s": float(finding[3]) * duration,
                    "end_s": float(finding[4]) * duration,
                    "evidence_frames": [int(value) for value in finding[5]],
                }
            )
        label_path = labels_root / f"{hashlib.sha256(identity.encode()).hexdigest()[:24]}.json"
        write_json(
            label_path,
            {
                "schema_version": "egoqc-visual-teacher-v1",
                "teacher_model": benchmark.get("model_id"),
                "prompt_version": prompt_version,
                "request_id": identity,
                "overall": {
                    "training_usable": None,
                    "recommended_route": "human_review",
                    "confidence": confidence,
                },
                "tasks": task_scores,
                "findings": findings,
                "summary": (
                    "未校准本地模型建议：" + "、".join(sorted(predicted_tasks))
                    if predicted_tasks
                    else "未校准本地模型未发现高于阈值的问题；仍需人工独立判断。"
                ),
                "label_role": "unscored_machine_suggestion_not_gold",
                "acceptance_authority": False,
                "training_label_authority": False,
                "probability_threshold": probability_threshold,
                "source_predictions_sha256": predictions_sha256,
            },
        )
        source_counts[str(request.get("source_class") or "unknown")] += 1
        rows.append(
            {
                **request,
                "output_path": str(label_path),
                "review_reason": "reserved_validation_local_vlm_suggestion",
                "machine_assessment_source": "local_few_b_vlm_unscored",
                "candidate_labels_are_not_gold": True,
                "accuracy_evaluation_eligible_after_human_adjudication_only": True,
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "review-queue.jsonl"
    write_jsonl(artifact, rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "queue": str(queue),
        "benchmark_root": str(benchmark_root),
        "review_requests": len(rows),
        "missing_predictions": missing,
        "invalid_predictions": invalid,
        "source_class_counts": dict(source_counts),
        "suggested_task_counts": dict(suggested_counts),
        "probability_threshold": probability_threshold,
        "machine_suggestions_are_gold": False,
        "may_auto_accept_or_reject": False,
        "may_train_before_human_adjudication": False,
        "raw_source_readonly": True,
        "code_version": code_version(),
        "artifact": str(artifact),
    }
    write_json(output / "summary.json", summary)
    return summary
