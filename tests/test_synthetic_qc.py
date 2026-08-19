import json
from pathlib import Path

import numpy as np

from egoqc.synthetic_qc import AUGMENTATIONS, build_synthetic_qc_training
from egoqc.vla_dataset import _apply_synthetic_augmentation


def test_build_synthetic_qc_training(tmp_path: Path) -> None:
    tasks = list(AUGMENTATIONS)
    row = {
        "record_id": "teacher:one",
        "video_id": "one",
        "source_uri": "/readonly/video.mp4",
        "provenance": {"raw_immutable": True},
        "vla_pretraining": {"split": "train"},
        "distillation": {
            "quality_band": "strong_negative",
            "split": "train",
            "tasks": tasks,
            "targets": {task: 0.0 for task in tasks},
            "label_masks": {task: 1 for task in tasks},
            "label_weights": {task: 0.4 for task in tasks},
            "label_sources": {task: "api_vlm_teacher" for task in tasks},
            "label_details": {task: {} for task in tasks},
        },
    }
    manifest = tmp_path / "source.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

    summary = build_synthetic_qc_training(manifest, tmp_path / "out", maximum_per_task=1)
    rows = [json.loads(line) for line in (tmp_path / "out/synthetic-train.jsonl").read_text().splitlines()]

    assert summary["synthetic_records"] == len(AUGMENTATIONS)
    assert summary["materialized_video_copies"] == 0
    assert {item["vla_pretraining"]["synthetic_augmentation"]["target_task"] for item in rows} == set(tasks)
    assert all(item["distillation"]["split"] == "train" for item in rows)


def test_all_synthetic_augmentations_change_frames() -> None:
    frames = np.zeros((8, 64, 64, 3), dtype=np.uint8)
    frames[:, 12:52, 12:52] = 180
    frames[:, 20:44, 20:44] = np.arange(8, dtype=np.uint8)[:, None, None, None] * 8
    for spec in AUGMENTATIONS.values():
        augmented = _apply_synthetic_augmentation(frames, {**spec, "seed": 7})
        assert augmented.shape == frames.shape
        assert not np.array_equal(augmented, frames)
