from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .provenance import code_version
from .report import write_json, write_jsonl
from .vla_dataset import VLAPretrainDataset, collate_vla_samples


SCHEMA_VERSION = "egoqc-qc-distillation-v1"
PROGRAMMATIC_TASKS = ("hand_absent", "persistent_extra_hands")


def _read_jsonl(path: Optional[Path]) -> Iterable[Dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
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


def _split(group_id: str) -> str:
    bucket = int(hashlib.sha256(("qc-distill:" + group_id).encode()).hexdigest()[:8], 16) % 1000
    if bucket < 800:
        return "train"
    if bucket < 900:
        return "validation"
    return "test"


def _split_group(
    source_row: Dict[str, Any], gold_row: Dict[str, Any]
) -> Tuple[str, str, str]:
    """Choose the strongest available identity boundary for leakage-safe splitting."""

    source_dataset = str(source_row.get("source_dataset") or "unknown-source")
    supplier = str(
        gold_row.get("supplier_id")
        or source_row.get("supplier_id")
        or source_row.get("vendor_id")
        or "unknown-supplier"
    )
    candidates = (
        ("person_id", gold_row.get("person_id") or source_row.get("person_id")),
        ("operator_id", gold_row.get("operator_id") or source_row.get("operator_id")),
        (
            "collection_session_id",
            gold_row.get("collection_session_id")
            or source_row.get("collection_session_id")
            or source_row.get("session_id"),
        ),
    )
    for source, value in candidates:
        if value not in (None, ""):
            return f"{source_dataset}:{supplier}:{source}:{value}", source, "low"
    upstream = source_row.get("vla_pretraining", {}).get("split_group")
    if upstream and str(upstream) != str(source_row.get("video_id")):
        return f"{source_dataset}:{supplier}:upstream:{upstream}", "upstream_split_group", "medium"
    video_id = str(source_row["video_id"])
    return f"{source_dataset}:{supplier}:video:{video_id}", "video_id_fallback", "high"


def _unique_by_video(rows: Sequence[Dict[str, Any]], source: str) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        video_id = str(row["video_id"])
        if video_id in result:
            raise ValueError(f"{source} 中 video_id 重复: {video_id}")
        result[video_id] = row
    return result


def _hand_report(root: Optional[Path], video_id: str) -> Optional[Dict[str, Any]]:
    if root is None:
        return None
    path = root / video_id / "hand-screen.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    value["_path"] = str(path)
    return value


def _programmatic_labels(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    metrics = report.get("metrics", {})
    gap = float(metrics.get("longest_no_hand_gap_s") or 0.0)
    extra_segments = metrics.get("suspected_extra_hand_segments") or []
    extra_ratio = float(metrics.get("suspected_extra_hands_ratio") or 0.0)
    # These are detector-derived soft targets, not human ground truth.
    absent_probability = min(1.0, max(0.0, (gap - 0.6) / 0.8))
    extra_probability = min(1.0, max(0.0, extra_ratio / 0.04)) if extra_segments else 0.0
    return {
        "hand_absent": {
            "probability": absent_probability,
            "confidence": 0.75,
            "source": "programmatic_hand_detector",
            "evidence": {"longest_no_hand_gap_s": gap},
        },
        "persistent_extra_hands": {
            "probability": extra_probability,
            "confidence": 0.7,
            "source": "programmatic_hand_detector",
            "evidence": {"suspected_extra_hands_ratio": extra_ratio, "segments": len(extra_segments)},
        },
    }


def _load_teacher(root: Optional[Path], video_id: str, tasks: set[str]) -> Dict[str, Dict[str, Any]]:
    if root is None:
        return {}
    path = root / video_id / "teacher-label.json"
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "egoqc-visual-teacher-v1":
        raise ValueError(f"教师标签 schema 不兼容: {path}")
    result = {}
    for task, label in value.get("tasks", {}).items():
        if task not in tasks:
            raise ValueError(f"教师输出未知 task={task}: {path}")
        probability = float(label["probability"])
        confidence = float(label.get("confidence", 0.0))
        if not 0 <= probability <= 1 or not 0 <= confidence <= 1:
            raise ValueError(f"教师 probability/confidence 越界: {path}")
        result[task] = {
            "probability": probability,
            "confidence": confidence,
            "source": "local_vlm_teacher",
            "teacher_model": value.get("teacher_model"),
            "prompt_version": value.get("prompt_version"),
            "artifact": str(path),
        }
    return result


def build_distillation_manifest(
    records: Path,
    task_config: Path,
    output: Path,
    *,
    hand_screen_root: Optional[Path] = None,
    teacher_root: Optional[Path] = None,
    gold_labels: Optional[Path] = None,
) -> Dict[str, Any]:
    config = json.loads(task_config.read_text(encoding="utf-8"))
    tasks = list(config["model_tasks"])
    task_set = set(tasks)
    gold_by_video = _unique_by_video(list(_read_jsonl(gold_labels)), "Gold Set")
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    source_counts = Counter()

    for source_row in _read_jsonl(records):
        if not source_row.get("vla_pretraining", {}).get("candidate"):
            continue
        video_id = str(source_row["video_id"])
        labels: Dict[str, Dict[str, Any]] = {}
        hand = _hand_report(hand_screen_root, video_id)
        if hand:
            labels.update(_programmatic_labels(hand))
        labels.update(_load_teacher(teacher_root, video_id, task_set))
        gold = gold_by_video.get(video_id, {})
        human_labels: Dict[str, Dict[str, Any]] = {}
        for task, raw_value in gold.get("labels", {}).items():
            if task not in task_set:
                raise ValueError(f"Gold Set 出现未知 task={task}, video_id={video_id}")
            human_labels[task] = {
                "probability": float(bool(raw_value)),
                "confidence": 1.0,
                "source": "human_gold",
                "reviewer_id": gold.get("reviewer_id"),
            }
        labels.update(human_labels)
        split_group, split_group_source, leakage_risk = _split_group(source_row, gold)
        split = _split(split_group)
        # Validation and test are measurement sets, never teacher self-evaluation.
        if split != "train":
            labels = human_labels
        if not labels:
            continue
        for label in labels.values():
            source_counts[label["source"]] += 1
        derived = dict(source_row)
        for field in (
            "source_revision",
            "source_dataset",
            "supplier_id",
            "person_id",
            "operator_id",
            "collection_session_id",
            "scene_id",
            "camera_id",
            "task_id",
        ):
            if gold.get(field) not in (None, ""):
                derived[field] = gold[field]
        derived["vla_pretraining"] = dict(source_row["vla_pretraining"])
        derived["vla_pretraining"]["candidate"] = True
        derived["vla_pretraining"]["split"] = split
        derived["vla_pretraining"]["split_group"] = split_group
        clip_start = gold.get("clip_start_s")
        clip_end = gold.get("clip_end_s")
        if clip_start is not None or clip_end is not None:
            if clip_start is None or clip_end is None:
                raise ValueError(f"Gold clip 必须同时提供 start/end: video_id={video_id}")
            clip_start = float(clip_start)
            clip_end = float(clip_end)
            duration = float(source_row.get("duration_s") or 0.0)
            if clip_start < 0 or clip_end <= clip_start or (duration > 0 and clip_end > duration + 1e-3):
                raise ValueError(
                    f"Gold clip 越界: video_id={video_id}, clip=[{clip_start},{clip_end}], duration={duration}"
                )
            derived["vla_pretraining"]["clip_sampler"] = {
                **source_row["vla_pretraining"]["clip_sampler"],
                "mode": "fixed_reviewed_window",
                "fixed_start_s": clip_start,
                "window_s": clip_end - clip_start,
            }
        derived["distillation"] = {
            "schema_version": SCHEMA_VERSION,
            "tasks": tasks,
            "split": split,
            "split_group": split_group,
            "split_group_source": split_group_source,
            "leakage_risk": leakage_risk,
            "label_scope": "reviewed_clip" if clip_start is not None else "video_level",
            "clip_start_s": clip_start,
            "clip_end_s": clip_end,
            "gold_provenance": {
                key: gold.get(key)
                for key in (
                    "reviewer_id",
                    "second_reviewer_id",
                    "adjudicator_id",
                    "reviewed_at",
                    "label_version",
                )
                if gold.get(key) is not None
            },
            "targets": {task: float(labels.get(task, {}).get("probability", 0.0)) for task in tasks},
            "label_masks": {task: int(task in labels) for task in tasks},
            "label_weights": {
                task: (
                    1.0 if labels.get(task, {}).get("source") == "human_gold"
                    else 0.5 * float(labels.get(task, {}).get("confidence", 0.0))
                    if labels.get(task, {}).get("source") == "local_vlm_teacher"
                    else 0.25 * float(labels.get(task, {}).get("confidence", 0.0))
                    if task in labels else 0.0
                )
                for task in tasks
            },
            "label_sources": {task: labels[task]["source"] for task in labels},
            "label_details": labels,
            "acceptance_authority": False,
            "evaluation_labels_are_human_only": split != "train",
        }
        manifest.append(derived)

    write_jsonl(output / "qc-distillation.jsonl", manifest)
    split_counts = Counter(row["distillation"]["split"] for row in manifest)
    task_counts = Counter(
        task for row in manifest for task, mask in row["distillation"]["label_masks"].items() if mask
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "records": len(manifest),
        "tasks": tasks,
        "split_counts": dict(split_counts),
        "task_label_counts": dict(task_counts),
        "label_source_counts": dict(source_counts),
        "split_group_source_counts": dict(
            Counter(row["distillation"]["split_group_source"] for row in manifest)
        ),
        "high_leakage_risk_records": sum(
            row["distillation"]["leakage_risk"] == "high" for row in manifest
        ),
        "gold_videos": len(gold_by_video),
        "calibration_status": "unavailable" if not gold_by_video else "requires_evaluation",
        "auto_reject_enabled": False,
        "code_version": code_version(),
        "manifest": str(output / "qc-distillation.jsonl"),
    }
    write_json(output / "summary.json", summary)
    return summary


def audit_qc_training_data(
    manifest: Path,
    task_config: Path,
    output: Path,
) -> Dict[str, Any]:
    """Audit a distillation manifest before any production student training."""

    rows = list(_read_jsonl(manifest))
    config = json.loads(task_config.read_text(encoding="utf-8"))
    tasks = list(config["model_tasks"])
    readiness = config.get("training_readiness", {})
    minimum_train = int(readiness.get("minimum_train_labeled_per_task", 500))
    minimum_train_gold = int(readiness.get("minimum_train_human_gold_per_task", 100))
    minimum_validation_positive = int(
        readiness.get("minimum_validation_gold_positives_per_task", 100)
    )
    minimum_validation_negative = int(
        readiness.get("minimum_validation_gold_negatives_per_task", 300)
    )
    minimum_test_positive = int(
        readiness.get("minimum_test_gold_positives_per_task", 200)
    )
    minimum_test_negative = int(
        readiness.get("minimum_test_gold_negatives_per_task", 1000)
    )
    minimum_safe_group_ratio = float(
        readiness.get("minimum_safe_split_group_ratio", 0.95)
    )

    blockers: List[Dict[str, Any]] = []
    coverage: Dict[str, Dict[str, Counter]] = {
        split: {task: Counter() for task in tasks}
        for split in ("train", "validation", "test")
    }
    split_groups: Dict[str, set[str]] = defaultdict(set)
    video_splits: Dict[str, set[str]] = defaultdict(set)
    uri_splits: Dict[str, set[str]] = defaultdict(set)
    supplier_splits: Dict[str, set[str]] = defaultdict(set)
    high_risk = 0
    invalid_rows = 0
    governance_blocked = 0
    metadata_missing_counts: Counter = Counter()

    for row_number, row in enumerate(rows, 1):
        distillation = row.get("distillation", {})
        video_id = str(row.get("video_id") or "")
        split = str(distillation.get("split") or "")
        group = str(distillation.get("split_group") or "")
        source_uri = str(row.get("source_uri") or "")
        if distillation.get("schema_version") != SCHEMA_VERSION or split not in coverage:
            invalid_rows += 1
            blockers.append({
                "code": "invalid_manifest_row",
                "row_number": row_number,
                "video_id": video_id,
            })
            continue
        if not video_id or not source_uri or not group:
            invalid_rows += 1
            blockers.append({
                "code": "missing_training_identity_or_source",
                "row_number": row_number,
                "video_id": video_id,
            })
            continue
        split_groups[group].add(split)
        video_splits[video_id].add(split)
        uri_splits[source_uri].add(split)
        supplier = row.get("supplier_id") or row.get("vendor_id")
        if supplier:
            supplier_splits[str(supplier)].add(split)
        for field in ("supplier_id", "scene_id", "camera_id", "task_id"):
            if not row.get(field):
                metadata_missing_counts[field] += 1
        if distillation.get("leakage_risk") == "high":
            high_risk += 1
        if split == "train" and not row.get("vla_pretraining", {}).get("training_ready", False):
            governance_blocked += 1

        targets = distillation.get("targets", {})
        masks = distillation.get("label_masks", {})
        sources = distillation.get("label_sources", {})
        for task in tasks:
            if not masks.get(task):
                continue
            source = str(sources.get(task) or "unknown")
            target = float(targets[task])
            stats = coverage[split][task]
            stats["labeled"] += 1
            stats["positive" if target >= 0.5 else "negative"] += 1
            stats[f"source:{source}"] += 1
            if source == "human_gold":
                stats["human_gold"] += 1
            elif split != "train":
                blockers.append({
                    "code": "weak_label_in_evaluation_split",
                    "video_id": video_id,
                    "split": split,
                    "task": task,
                    "source": source,
                })
        if split != "train" and not distillation.get("label_details"):
            blockers.append({
                "code": "evaluation_provenance_missing",
                "video_id": video_id,
                "split": split,
            })
        if split != "train":
            provenance = distillation.get("gold_provenance", {})
            missing_provenance = [
                key
                for key in ("reviewer_id", "reviewed_at", "label_version")
                if not provenance.get(key)
            ]
            if missing_provenance:
                blockers.append({
                    "code": "evaluation_gold_provenance_incomplete",
                    "video_id": video_id,
                    "split": split,
                    "missing": missing_provenance,
                })
        if split != "train" and (
            distillation.get("clip_start_s") is None
            or distillation.get("clip_end_s") is None
        ):
            blockers.append({
                "code": "evaluation_clip_window_missing",
                "video_id": video_id,
                "split": split,
            })

    for code, values in (
        ("split_group_leakage", split_groups),
        ("video_id_leakage", video_splits),
        ("source_uri_leakage", uri_splits),
    ):
        for identity, splits in values.items():
            if len(splits) > 1:
                blockers.append({
                    "code": code,
                    "identity": identity,
                    "splits": sorted(splits),
                })

    task_reports: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        train = coverage["train"][task]
        validation = coverage["validation"][task]
        test = coverage["test"][task]
        requirements = {
            "train_labeled": (train["labeled"], minimum_train),
            "train_human_gold": (train["human_gold"], minimum_train_gold),
            "validation_gold_positive": (validation["positive"], minimum_validation_positive),
            "validation_gold_negative": (validation["negative"], minimum_validation_negative),
            "test_gold_positive": (test["positive"], minimum_test_positive),
            "test_gold_negative": (test["negative"], minimum_test_negative),
        }
        task_blockers = [
            name for name, (actual, required) in requirements.items() if actual < required
        ]
        task_reports[task] = {
            "ready": not task_blockers,
            "block_reasons": task_blockers,
            "requirements": {
                name: {"actual": actual, "required": required}
                for name, (actual, required) in requirements.items()
            },
            "coverage": {
                split: dict(coverage[split][task])
                for split in ("train", "validation", "test")
            },
        }

    safe_group_ratio = 1.0 - high_risk / len(rows) if rows else 0.0
    global_block_reasons = []
    if not rows:
        global_block_reasons.append("empty_manifest")
    if invalid_rows:
        global_block_reasons.append("invalid_manifest_rows")
    if any(item["code"].endswith("leakage") for item in blockers):
        global_block_reasons.append("cross_split_leakage")
    if any(item["code"] == "weak_label_in_evaluation_split" for item in blockers):
        global_block_reasons.append("weak_evaluation_labels")
    if safe_group_ratio < minimum_safe_group_ratio:
        global_block_reasons.append("insufficient_person_or_session_metadata")
    if governance_blocked:
        global_block_reasons.append("training_license_or_governance_missing")
    if metadata_missing_counts:
        global_block_reasons.append("evaluation_slice_metadata_incomplete")
    evaluation_contract_codes = {
        "evaluation_provenance_missing",
        "evaluation_gold_provenance_incomplete",
        "evaluation_clip_window_missing",
    }
    if any(item["code"] in evaluation_contract_codes for item in blockers):
        global_block_reasons.append("evaluation_contract_incomplete")
    if not all(value["ready"] for value in task_reports.values()):
        global_block_reasons.append("task_coverage_incomplete")

    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "qc-training-blockers.jsonl", blockers)
    report = {
        "schema_version": "egoqc-qc-training-readiness-v1",
        "ready_for_production_training": not global_block_reasons,
        "records": len(rows),
        "invalid_rows": invalid_rows,
        "safe_split_group_ratio": safe_group_ratio,
        "minimum_safe_split_group_ratio": minimum_safe_group_ratio,
        "governance_blocked_train_records": governance_blocked,
        "split_counts": dict(Counter(row.get("distillation", {}).get("split") for row in rows)),
        "split_group_count": len(split_groups),
        "supplier_count": len(supplier_splits),
        "metadata_missing_counts": dict(metadata_missing_counts),
        "global_block_reasons": global_block_reasons,
        "tasks": task_reports,
        "blocker_count": len(blockers),
        "blockers": str(output / "qc-training-blockers.jsonl"),
        "manifest": str(manifest.expanduser().resolve()),
        "code_version": code_version(),
    }
    write_json(output / "qc-training-readiness.json", report)
    return report


def smoke_train_qc_student(
    manifest: Path,
    output: Path,
    *,
    steps: int = 20,
    batch_size: int = 4,
    device: str = "cuda",
    learning_rate: float = 5e-4,
    seed: int = 0,
) -> Dict[str, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as error:
        raise RuntimeError("QC student 训练需要 torch") from error
    torch.manual_seed(seed)
    dataset = VLAPretrainDataset(
        manifest, split="train", allow_technical_candidates=True, seed=seed
    )
    if len(dataset) < 2:
        raise ValueError(f"训练样本不足: {len(dataset)}")
    batch_size = min(batch_size, len(dataset))
    samples = [dataset[index] for index in range(batch_size)]
    batch = collate_vla_samples(samples)
    distillation = [sample["distillation"] for sample in samples]
    tasks = distillation[0]["tasks"]
    frames = torch.from_numpy(batch["frames"]).to(device=device, dtype=torch.float32)
    frames = frames.permute(0, 1, 4, 2, 3)[:, ::4].contiguous().div_(255.0)
    targets = torch.tensor(
        [[item["targets"][task] for task in tasks] for item in distillation],
        dtype=torch.float32, device=device,
    )
    weights = torch.tensor(
        [[item["label_weights"][task] * item["label_masks"][task] for task in tasks] for item in distillation],
        dtype=torch.float32, device=device,
    )
    label_stats = {}
    for task_index, task in enumerate(tasks):
        observed = weights[:, task_index] > 0
        values = targets[:, task_index][observed]
        label_stats[task] = {
            "labeled": int(observed.sum().item()),
            "positive_soft_ge_0_5": int((values >= 0.5).sum().item()),
            "negative_soft_lt_0_5": int((values < 0.5).sum().item()),
        }
    active_tasks = {task for task, stats in label_stats.items() if stats["labeled"] > 0}

    class TemporalQCStudent(nn.Module):
        def __init__(self, outputs: int) -> None:
            super().__init__()
            self.frame_encoder = nn.Sequential(
                nn.Conv2d(3, 24, 7, stride=4, padding=3), nn.GELU(),
                nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.temporal = nn.GRU(48, 64, batch_first=True, bidirectional=True)
            self.head = nn.Linear(128, outputs)

        def forward(self, video):
            batch_n, time_n = video.shape[:2]
            encoded = self.frame_encoder(video.reshape(batch_n * time_n, *video.shape[2:])).flatten(1)
            sequence, _ = self.temporal(encoded.reshape(batch_n, time_n, -1))
            return self.head(sequence.mean(1))

    model = TemporalQCStudent(len(tasks)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = []
    started = time.perf_counter()
    use_bfloat16 = device.startswith("cuda") and torch.cuda.is_bf16_supported()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
            logits = model(frames)
            per_label = functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            loss = (per_label * weights).sum() / weights.sum().clamp_min(1e-6)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        history.append({
            "step": step + 1,
            "loss": float(loss.detach().cpu()),
            "gradient_norm": float(gradient_norm.detach().cpu()),
        })
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    probabilities = torch.sigmoid(model(frames).detach()).cpu().numpy()
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "qc-student-smoke.pt"
    torch.save({"model": model.state_dict(), "tasks": tasks, "schema_version": SCHEMA_VERSION}, checkpoint)
    report = {
        "status": "succeeded",
        "purpose": "distillation engineering smoke; not calibrated for acceptance",
        "samples": len(samples),
        "tasks": tasks,
        "active_tasks": sorted(active_tasks),
        "inactive_tasks": [task for task in tasks if task not in active_tasks],
        "label_stats": label_stats,
        "steps": steps,
        "history": history,
        "elapsed_s": elapsed,
        "gpu_name": torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "predictions": [
            {
                "video_id": samples[index]["video_id"],
                "probabilities": {
                    task: float(probabilities[index, task_index]) if task in active_tasks else None
                    for task_index, task in enumerate(tasks)
                },
            }
            for index in range(len(samples))
        ],
        "calibrated": False,
        "auto_reject_enabled": False,
        "checkpoint": str(checkpoint),
    }
    write_json(output / "qc-student-smoke.json", report)
    return report


def _wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    probability = successes / total
    denominator = 1.0 + z * z / total
    centre = probability + z * z / (2.0 * total)
    margin = z * math.sqrt(
        probability * (1.0 - probability) / total
        + z * z / (4.0 * total * total)
    )
    return max(0.0, (centre - margin) / denominator)


def evaluate_qc_predictions(
    predictions: Path,
    gold_labels: Path,
    task_config: Path,
    output: Path,
) -> Dict[str, Any]:
    config = json.loads(task_config.read_text(encoding="utf-8"))
    tasks = config["model_tasks"]
    policy = config["policy"]
    predicted = {str(row["video_id"]): row for row in _read_jsonl(predictions)}
    gold = {str(row["video_id"]): row for row in _read_jsonl(gold_labels)}
    minimum_positives = int(policy["minimum_gold_positives_per_task"])
    minimum_negatives = int(policy["minimum_gold_negatives_per_task"])
    task_reports = {}

    for task, task_config_value in tasks.items():
        pairs = []
        for video_id, gold_row in gold.items():
            if task not in gold_row.get("labels", {}) or video_id not in predicted:
                continue
            raw_probability = predicted[video_id].get("probabilities", {}).get(task)
            if raw_probability is None:
                continue
            probability = float(raw_probability)
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(f"预测概率越界: video_id={video_id}, task={task}")
            pairs.append((probability, int(bool(gold_row["labels"][task]))))
        positives = sum(label for _, label in pairs)
        negatives = len(pairs) - positives
        target_precision = float(task_config_value["minimum_auto_reject_precision"])
        candidates = []
        for threshold_index in range(5, 100):
            threshold = threshold_index / 100.0
            tp = sum(probability >= threshold and label == 1 for probability, label in pairs)
            fp = sum(probability >= threshold and label == 0 for probability, label in pairs)
            fn = sum(probability < threshold and label == 1 for probability, label in pairs)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            if tp > 0 and precision >= target_precision:
                candidates.append((recall, precision, threshold, tp, fp))
        best = max(candidates, default=None, key=lambda value: (value[0], value[1], value[2]))
        brier = (
            sum((probability - label) ** 2 for probability, label in pairs) / len(pairs)
            if pairs else None
        )
        ece = 0.0
        if pairs:
            for bin_index in range(10):
                lower, upper = bin_index / 10, (bin_index + 1) / 10
                bucket = [pair for pair in pairs if lower <= pair[0] < upper or (upper == 1 and pair[0] == 1)]
                if not bucket:
                    continue
                confidence = sum(probability for probability, _ in bucket) / len(bucket)
                accuracy = sum(label for _, label in bucket) / len(bucket)
                ece += len(bucket) / len(pairs) * abs(confidence - accuracy)
        enough_gold = positives >= minimum_positives and negatives >= minimum_negatives
        precision_lower_bound = (
            _wilson_lower_bound(best[3], best[3] + best[4]) if best is not None else None
        )
        statistically_supported = (
            precision_lower_bound is not None and precision_lower_bound >= target_precision
        )
        enabled = enough_gold and best is not None and statistically_supported
        task_reports[task] = {
            "gold_samples": len(pairs),
            "gold_positives": positives,
            "gold_negatives": negatives,
            "required_positives": minimum_positives,
            "required_negatives": minimum_negatives,
            "target_precision": target_precision,
            "precision_confidence_method": "wilson_two_sided_95_percent_lower_bound",
            "brier_score": brier,
            "ece_10_bin": ece if pairs else None,
            "auto_reject_enabled": enabled,
            "threshold": best[2] if enabled else None,
            "precision": best[1] if enabled else None,
            "empirical_precision": best[1] if best is not None else None,
            "precision_95_lower_bound": precision_lower_bound,
            "recall": best[0] if enabled else None,
            "block_reasons": [
                reason for condition, reason in (
                    (not enough_gold, "insufficient_gold_coverage"),
                    (best is None, "precision_target_not_reached"),
                    (
                        best is not None and not statistically_supported,
                        "precision_confidence_bound_below_target",
                    ),
                ) if condition
            ],
        }

    report = {
        "schema_version": "egoqc-qc-evaluation-v1",
        "predictions": str(predictions.expanduser().resolve()),
        "gold_labels": str(gold_labels.expanduser().resolve()),
        "tasks": task_reports,
        "enabled_tasks": [task for task, value in task_reports.items() if value["auto_reject_enabled"]],
        "rule_failures_remain_authoritative": True,
    }
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "qc-student-evaluation.json", report)
    return report
