from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .provenance import code_version
from .report import write_json, write_jsonl


SCHEMA_VERSION = "egoqc-teacher-training-pool-v1"


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def build_teacher_training_pool(
    queue: Path,
    task_config: Path,
    output: Path,
    *,
    minimum_overall_confidence: float = 0.85,
    minimum_task_confidence: float = 0.70,
    strong_positive_probability: float = 0.80,
    strong_negative_probability: float = 0.20,
) -> Dict[str, Any]:
    """Build train-only weak-label manifests from a teacher API queue.

    Teacher labels never become validation/test truth. The high-confidence view
    keeps only unambiguous positive or negative clips; the complete view retains
    soft boundary examples for later curriculum training.
    """

    for value in (
        minimum_overall_confidence,
        minimum_task_confidence,
        strong_positive_probability,
        strong_negative_probability,
    ):
        if not 0 <= value <= 1:
            raise ValueError("概率与置信度阈值必须位于 [0, 1]")
    config = json.loads(task_config.read_text(encoding="utf-8"))
    tasks = list(config["model_tasks"])
    task_set = set(tasks)
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    high_rows: List[Dict[str, Any]] = []
    skipped = Counter()
    bands = Counter()
    positives = Counter()

    for request in _read_jsonl(queue):
        label_path = Path(request["output_path"])
        if not label_path.is_file():
            skipped["missing_label"] += 1
            continue
        label = json.loads(label_path.read_text(encoding="utf-8"))
        if label.get("schema_version") != "egoqc-visual-teacher-v1":
            skipped["invalid_schema"] += 1
            continue
        overall = label.get("overall") or {}
        route = str(overall.get("recommended_route") or "")
        overall_confidence = float(overall.get("confidence") or 0.0)
        if route == "human_review":
            skipped["teacher_requested_human_review"] += 1
            continue
        if overall_confidence < minimum_overall_confidence:
            skipped["low_overall_confidence"] += 1
            continue

        raw_tasks = label.get("tasks") or {}
        unknown = set(raw_tasks) - task_set
        if unknown:
            skipped["unknown_task"] += 1
            continue
        probabilities = {
            task: float((raw_tasks.get(task) or {}).get("probability", 0.0))
            for task in tasks
        }
        confidences = {
            task: float((raw_tasks.get(task) or {}).get("confidence", 0.0))
            for task in tasks
        }
        masks = {task: int(confidences[task] >= minimum_task_confidence) for task in tasks}
        if not any(masks.values()):
            skipped["no_confident_task_labels"] += 1
            continue

        strong_tasks = [
            task for task in tasks
            if masks[task] and probabilities[task] >= strong_positive_probability
        ]
        strong_negative = (
            route == "accept"
            and bool(overall.get("training_usable"))
            and all(
                not masks[task] or probabilities[task] <= strong_negative_probability
                for task in tasks
            )
        )
        if strong_tasks:
            band = "strong_positive"
        elif strong_negative:
            band = "strong_negative"
        else:
            band = "soft_boundary"
        bands[band] += 1
        positives.update(strong_tasks)

        request_id = str(request["request_id"])
        start_s = float(request["clip_start_s"])
        end_s = float(request["clip_end_s"])
        source_uri = str(request.get("raw_source_uri") or request["source_uri"])
        teacher_weight_cap = 0.5
        row = {
            "record_id": f"teacher:{request_id}",
            "video_id": request_id,
            "source_class": "public_dataset",
            "source_dataset": "egodex",
            "source_uri": source_uri,
            "duration_s": end_s,
            "activities": [str(request.get("task") or "")],
            "candidate_tier": request.get("candidate_tier"),
            "provenance": {
                "raw_immutable": True,
                "teacher_artifact": str(label_path),
                "teacher_model": label.get("teacher_model"),
                "prompt_version": label.get("prompt_version"),
                "code_version": code_version(),
            },
            "vla_pretraining": {
                "candidate": True,
                "training_ready": True,
                "split": "train",
                "split_group": request_id,
                "split_warning": "teacher_weak_label_train_only",
                "allowed_objectives": ["qc_visual_semantics"],
                "loss_masks": {
                    "video_representation": 0,
                    "temporal_prediction": 0,
                    "video_text_alignment": 0,
                    "hand_presence_auxiliary": 0,
                    "mano_motion": 0,
                    "robot_action": 0,
                    "camera_pose": 0,
                    "tactile": 0,
                    "qc_visual_semantics": 1,
                },
                "target_availability": {"qc_visual_semantics": True},
                "clip_sampler": {
                    "mode": "fixed_reviewed_window",
                    "fixed_start_s": start_s,
                    "window_s": end_s - start_s,
                    "minimum_visible_duration_s": 0.0,
                    "decode_fps": 8.0,
                },
            },
            "distillation": {
                "schema_version": "egoqc-qc-distillation-v1",
                "tasks": tasks,
                "split": "train",
                "split_group": request_id,
                "split_group_source": "request_id",
                "leakage_risk": "high",
                "label_scope": "reviewed_clip",
                "clip_start_s": start_s,
                "clip_end_s": end_s,
                "quality_band": band,
                "targets": probabilities,
                "label_masks": masks,
                "label_weights": {
                    task: (
                        teacher_weight_cap * overall_confidence * confidences[task]
                        if masks[task] else 0.0
                    )
                    for task in tasks
                },
                "label_sources": {
                    task: "api_vlm_teacher" for task in tasks if masks[task]
                },
                "label_details": {
                    task: {
                        "probability": probabilities[task],
                        "confidence": confidences[task],
                        "source": "api_vlm_teacher",
                    }
                    for task in tasks if masks[task]
                },
                "acceptance_authority": False,
                "evaluation_labels_are_human_only": True,
            },
        }
        all_rows.append(row)
        if band != "soft_boundary":
            high_rows.append(row)

    write_jsonl(output / "teacher-train-all.jsonl", all_rows)
    write_jsonl(output / "teacher-train-high-confidence.jsonl", high_rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "queue": str(queue),
        "records": len(all_rows),
        "high_confidence_records": len(high_rows),
        "quality_bands": dict(bands),
        "strong_positive_task_counts": dict(positives),
        "skipped": dict(skipped),
        "governance": {
            "labels_are_weak": True,
            "train_only": True,
            "validation_and_test_require_human_gold": True,
            "may_auto_reject": False,
            "raw_source_readonly": True,
        },
        "artifacts": {
            "all": str(output / "teacher-train-all.jsonl"),
            "high_confidence": str(output / "teacher-train-high-confidence.jsonl"),
        },
    }
    write_json(output / "summary.json", summary)
    return summary
