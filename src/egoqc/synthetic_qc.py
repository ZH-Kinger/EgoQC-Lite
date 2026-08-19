from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .provenance import code_version
from .report import write_json, write_jsonl


SCHEMA_VERSION = "egoqc-synthetic-qc-training-v1"
AUGMENTATIONS: Dict[str, Dict[str, Any]] = {
    "semantic_camera_shake": {
        "kind": "camera_shake",
        "probability": 0.95,
        "max_translation_ratio": 0.055,
        "max_rotation_deg": 4.0,
    },
    "frozen_or_duplicate_frames": {
        "kind": "freeze_segment",
        "probability": 1.0,
        "start_fraction": 0.30,
        "duration_fraction": 0.45,
    },
    "unusable_visual_quality": {
        "kind": "blur_downsample",
        "probability": 0.95,
        "downsample_scale": 0.14,
        "gaussian_radius": 3.2,
    },
    "severe_occlusion": {
        "kind": "foreground_occlusion",
        "probability": 0.90,
        "width_fraction": 0.52,
        "height_fraction": 0.42,
        "center_x_fraction": 0.50,
        "center_y_fraction": 0.72,
    },
}


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _rank(video_id: str, task: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{task}:{video_id}".encode()).hexdigest()


def build_synthetic_qc_training(
    manifest: Path,
    output: Path,
    *,
    maximum_per_task: int = 400,
    seed: int = 31,
) -> Dict[str, Any]:
    if maximum_per_task < 1:
        raise ValueError("maximum_per_task 必须 >= 1")
    source_rows = [
        row for row in _read_jsonl(manifest)
        if row.get("distillation", {}).get("quality_band") == "strong_negative"
        and row.get("distillation", {}).get("split") == "train"
    ]
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    synthetic: List[Dict[str, Any]] = []
    counts = Counter()

    for task, spec in AUGMENTATIONS.items():
        eligible = [
            row for row in source_rows
            if task in row.get("distillation", {}).get("tasks", [])
        ]
        eligible.sort(key=lambda row: _rank(str(row["video_id"]), task, seed))
        for source in eligible[:maximum_per_task]:
            row = copy.deepcopy(source)
            source_video_id = str(source["video_id"])
            variant_id = hashlib.sha256(
                f"{SCHEMA_VERSION}:{seed}:{task}:{source_video_id}".encode()
            ).hexdigest()[:16]
            row["record_id"] = f"synthetic:{variant_id}"
            row["video_id"] = f"synthetic-{variant_id}"
            row["source_class"] = "derived_public_training_view"
            row["synthetic_source_video_id"] = source_video_id
            row["provenance"] = {
                **row.get("provenance", {}),
                "synthetic": True,
                "synthetic_schema_version": SCHEMA_VERSION,
                "synthetic_parent_video_id": source_video_id,
                "code_version": code_version(),
                "raw_immutable": True,
            }
            row["vla_pretraining"]["synthetic_augmentation"] = {
                **spec,
                "target_task": task,
                "seed": int(variant_id[:8], 16),
                "materialized": False,
            }
            distillation = row["distillation"]
            distillation["quality_band"] = "synthetic_strong_positive"
            distillation["synthetic_parent_video_id"] = source_video_id
            distillation["targets"][task] = float(spec["probability"])
            distillation["label_masks"][task] = 1
            distillation["label_weights"][task] = 0.35
            distillation["label_sources"][task] = "synthetic_controlled"
            distillation["label_details"][task] = {
                "probability": float(spec["probability"]),
                "confidence": 1.0,
                "source": "synthetic_controlled",
                "augmentation_kind": spec["kind"],
            }
            distillation["acceptance_authority"] = False
            distillation["evaluation_labels_are_human_only"] = True
            synthetic.append(row)
            counts[task] += 1

    combined = list(_read_jsonl(manifest)) + synthetic
    write_jsonl(output / "synthetic-train.jsonl", synthetic)
    write_jsonl(output / "combined-train.jsonl", combined)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "source_manifest": str(manifest),
        "source_strong_negatives": len(source_rows),
        "synthetic_records": len(synthetic),
        "combined_records": len(combined),
        "counts_by_target": dict(counts),
        "maximum_per_task": maximum_per_task,
        "materialized_video_copies": 0,
        "governance": {
            "train_only": True,
            "synthetic_labels_are_not_gold": True,
            "may_auto_reject": False,
            "raw_source_readonly": True,
        },
        "artifacts": {
            "synthetic": str(output / "synthetic-train.jsonl"),
            "combined": str(output / "combined-train.jsonl"),
        },
    }
    write_json(output / "summary.json", summary)
    return summary
