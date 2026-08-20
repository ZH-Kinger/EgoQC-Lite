from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .provenance import code_version
from .report import write_json, write_jsonl


SCHEMA_VERSION = "egoqc-local-vlm-training-pool-v1"


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.expanduser().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: each row must be an object")
            yield row


def _split_group(request: Dict[str, Any], source_dataset: str, source_uri: str) -> Tuple[str, str, str]:
    if request.get("split_group"):
        return (
            str(request["split_group"]),
            str(request.get("split_group_source") or "upstream_split_group"),
            str(request.get("leakage_risk") or "medium"),
        )
    digest = hashlib.sha256(source_uri.encode("utf-8")).hexdigest()[:20]
    if source_uri:
        return f"{source_dataset}:raw-video:{digest}", "raw_source_uri", "low"
    return str(request["request_id"]), "request_id_fallback", "high"


def _has_mano_overlay(request: Dict[str, Any]) -> bool:
    explicit = (
        "mano_overlay_uri",
        "overlay_uri",
        "annotated_video_uri",
        "visualization_uri",
    )
    return any(request.get(key) for key in explicit)


def build_local_vlm_training_pool(
    queue: Path,
    predictions: Path,
    task_config: Path,
    output: Path,
    *,
    minimum_overall_confidence: float = 0.80,
    strong_positive_probability: float = 0.80,
    strong_negative_probability: float = 0.20,
    local_teacher_weight_cap: float = 0.25,
) -> Dict[str, Any]:
    """Convert compact local few-B predictions into train-only weak labels.

    The local model is never treated as acceptance truth. Invalid, uncertain,
    abstained, and deterministic/model-disagreement clips are emitted to a
    separate review queue. MANO overlay drift is masked unless the request
    explicitly identifies an overlay artifact.
    """

    for value in (
        minimum_overall_confidence,
        strong_positive_probability,
        strong_negative_probability,
        local_teacher_weight_cap,
    ):
        if not 0 <= value <= 1:
            raise ValueError("probabilities, confidence and weights must be in [0, 1]")

    config = json.loads(task_config.expanduser().read_text(encoding="utf-8"))
    tasks = list(config["model_tasks"])
    task_set = set(tasks)
    prediction_rows = list(_read_jsonl(predictions))
    prediction_by_id: Dict[str, Dict[str, Any]] = {}
    duplicate_predictions = 0
    for prediction in prediction_rows:
        identity = str(prediction.get("video_id") or prediction.get("record_id") or "")
        if not identity:
            continue
        if identity in prediction_by_id:
            duplicate_predictions += 1
        prediction_by_id[identity] = prediction

    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, Any]] = []
    high_rows: List[Dict[str, Any]] = []
    review_rows: List[Dict[str, Any]] = []
    skipped = Counter()
    bands = Counter()
    positives = Counter()
    source_counts = Counter()

    for request in _read_jsonl(queue):
        request_id = str(request.get("request_id") or request.get("video_id") or "")
        prediction = prediction_by_id.get(request_id)
        if prediction is None:
            skipped["missing_prediction"] += 1
            review_rows.append({**request, "review_reason": "missing_local_prediction"})
            continue
        if not prediction.get("structured_json_valid"):
            skipped["invalid_structured_output"] += 1
            review_rows.append(
                {
                    **request,
                    "review_reason": "invalid_local_structured_output",
                    "local_prediction": prediction,
                }
            )
            continue
        parsed = prediction.get("parsed_response") or {}
        if parsed.get("a") is True:
            skipped["model_abstained"] += 1
            review_rows.append(
                {**request, "review_reason": "local_model_abstained", "local_prediction": prediction}
            )
            continue
        try:
            overall_confidence = float(parsed["c"])
        except (KeyError, TypeError, ValueError):
            skipped["invalid_confidence"] += 1
            review_rows.append(
                {**request, "review_reason": "invalid_local_confidence", "local_prediction": prediction}
            )
            continue
        if not 0 <= overall_confidence <= 1:
            skipped["invalid_confidence"] += 1
            continue
        if overall_confidence < minimum_overall_confidence:
            skipped["low_overall_confidence"] += 1
            review_rows.append(
                {
                    **request,
                    "review_reason": "low_local_confidence",
                    "local_prediction": prediction,
                }
            )
            continue

        findings = parsed.get("f")
        if not isinstance(findings, list):
            skipped["invalid_findings"] += 1
            continue
        probabilities = {task: 0.0 for task in tasks}
        details: Dict[str, Dict[str, Any]] = {}
        invalid_finding = False
        for finding in findings:
            if not isinstance(finding, list) or len(finding) != 6:
                invalid_finding = True
                break
            task = str(finding[0])
            if task not in task_set:
                invalid_finding = True
                break
            try:
                probability = float(finding[1])
                severity = int(finding[2])
                start_fraction = float(finding[3])
                end_fraction = float(finding[4])
                evidence_frames = [int(value) for value in finding[5]]
            except (TypeError, ValueError):
                invalid_finding = True
                break
            if not 0 <= probability <= 1 or not 0 <= severity <= 3:
                invalid_finding = True
                break
            probabilities[task] = max(probabilities[task], probability)
            details[task] = {
                "probability": probability,
                "confidence": overall_confidence,
                "severity": severity,
                "start_fraction": start_fraction,
                "end_fraction": end_fraction,
                "evidence_frames": evidence_frames,
                "source": "local_few_b_vlm_teacher",
            }
        if invalid_finding:
            skipped["invalid_findings"] += 1
            review_rows.append(
                {**request, "review_reason": "invalid_local_findings", "local_prediction": prediction}
            )
            continue

        masks = {task: 1 for task in tasks}
        if "mano_overlay_drift" in masks and not _has_mano_overlay(request):
            masks["mano_overlay_drift"] = 0
        positive_tasks = [
            task for task in tasks
            if masks[task] and probabilities[task] >= strong_positive_probability
        ]
        strong_negative = all(
            not masks[task] or probabilities[task] <= strong_negative_probability
            for task in tasks
        )
        selection_source = str(request.get("selection_source") or "")
        deterministic_bad = selection_source == "deterministic_bad_frame"
        deterministic_control = selection_source in {
            "deterministic_clean_gap_control",
            "deterministic_low_event_control",
        }
        disagreement = (deterministic_bad and not positive_tasks) or (
            deterministic_control and bool(positive_tasks)
        )
        if disagreement:
            band = "deterministic_model_disagreement"
        elif positive_tasks:
            band = "strong_positive"
        elif strong_negative:
            band = "strong_negative"
        else:
            band = "soft_boundary"
        bands[band] += 1
        positives.update(positive_tasks)

        source_uri = str(request.get("raw_source_uri") or request.get("source_uri") or "")
        source_dataset = str(request.get("source_dataset") or "unknown_dataset")
        split_group, split_group_source, leakage_risk = _split_group(
            request, source_dataset, source_uri
        )
        start_s = float(request.get("clip_start_s") or 0.0)
        end_s = float(request.get("clip_end_s") or start_s)
        activities = list(request.get("tasks") or [str(request.get("task") or "")])
        source_counts[source_dataset] += 1
        row = {
            "record_id": f"local-teacher:{request_id}",
            "video_id": request_id,
            "source_class": request.get("source_class") or "unknown",
            "source_dataset": source_dataset,
            "supplier_id": request.get("supplier_id"),
            "parent_episode_index": request.get("parent_episode_index"),
            "parent_episode_id": request.get("episode_id"),
            "scene_id": request.get("scene_id"),
            "camera_id": request.get("camera_id"),
            "task_id": request.get("task_id") or next((value for value in activities if value), None),
            "source_uri": source_uri,
            "duration_s": float(request.get("duration_s") or end_s),
            "activities": activities,
            "candidate_tier": request.get("candidate_tier"),
            "selection_source": request.get("selection_source"),
            "event_codes": list(request.get("event_codes") or []),
            "provenance": {
                "raw_immutable": True,
                "local_prediction_artifact": str(predictions.expanduser().resolve()),
                "local_model_prediction_schema": prediction.get("schema_version"),
                "frame_count": prediction.get("frame_count"),
                "maximum_edge": prediction.get("maximum_edge"),
                "code_version": code_version(),
            },
            "vla_pretraining": {
                "candidate": True,
                "training_ready": True,
                "split": "train",
                "split_group": split_group,
                "split_warning": "local_vlm_weak_label_train_only",
                "allowed_objectives": ["qc_visual_semantics"],
                "loss_masks": {"qc_visual_semantics": 1},
                "target_availability": {"qc_visual_semantics": True},
                "clip_sampler": {
                    "mode": "fixed_reviewed_window",
                    "fixed_start_s": start_s,
                    "window_s": max(0.0, end_s - start_s),
                    "minimum_visible_duration_s": 0.0,
                    "decode_fps": 8.0,
                },
            },
            "distillation": {
                "schema_version": "egoqc-qc-distillation-v1",
                "tasks": tasks,
                "split": "train",
                "split_group": split_group,
                "split_group_source": split_group_source,
                "leakage_risk": leakage_risk,
                "label_scope": "sampled_clip_frames",
                "clip_start_s": start_s,
                "clip_end_s": end_s,
                "quality_band": band,
                "targets": probabilities,
                "label_masks": masks,
                "label_weights": {
                    task: local_teacher_weight_cap * overall_confidence if masks[task] else 0.0
                    for task in tasks
                },
                "label_sources": {
                    task: "local_few_b_vlm_teacher" for task in tasks if masks[task]
                },
                "label_details": {
                    task: details.get(
                        task,
                        {
                            "probability": 0.0,
                            "confidence": overall_confidence,
                            "source": "local_few_b_vlm_teacher",
                        },
                    )
                    for task in tasks
                    if masks[task]
                },
                "acceptance_authority": False,
                "evaluation_labels_are_human_only": True,
            },
        }
        all_rows.append(row)
        if disagreement:
            review_rows.append(
                {
                    **request,
                    "review_reason": "deterministic_local_model_disagreement",
                    "local_prediction": prediction,
                }
            )
        elif band in {"strong_positive", "strong_negative"}:
            high_rows.append(row)

    write_jsonl(output / "local-teacher-train-all.jsonl", all_rows)
    write_jsonl(output / "local-teacher-train-high-confidence.jsonl", high_rows)
    write_jsonl(output / "local-teacher-human-review.jsonl", review_rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "queue": str(queue.expanduser().resolve()),
        "predictions": str(predictions.expanduser().resolve()),
        "prediction_rows": len(prediction_rows),
        "duplicate_prediction_ids": duplicate_predictions,
        "records": len(all_rows),
        "high_confidence_records": len(high_rows),
        "human_review_records": len(review_rows),
        "quality_bands": dict(bands),
        "strong_positive_task_counts": dict(positives),
        "source_counts": dict(source_counts),
        "skipped": dict(skipped),
        "governance": {
            "labels_are_weak": True,
            "train_only": True,
            "validation_and_test_require_human_gold": True,
            "may_auto_reject": False,
            "raw_source_readonly": True,
            "mano_overlay_drift_requires_explicit_overlay": True,
            "local_teacher_weight_cap": local_teacher_weight_cap,
        },
        "artifacts": {
            "all": str(output / "local-teacher-train-all.jsonl"),
            "high_confidence": str(output / "local-teacher-train-high-confidence.jsonl"),
            "human_review": str(output / "local-teacher-human-review.jsonl"),
        },
    }
    write_json(output / "summary.json", summary)
    return summary
