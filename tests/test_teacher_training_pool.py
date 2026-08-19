import json
from pathlib import Path

from egoqc.teacher_training_pool import build_teacher_training_pool


def test_build_teacher_training_pool_is_train_only(tmp_path: Path) -> None:
    config = tmp_path / "tasks.json"
    config.write_text(json.dumps({"model_tasks": {"shake": {}}}), encoding="utf-8")
    label = tmp_path / "label.json"
    label.write_text(json.dumps({
        "schema_version": "egoqc-visual-teacher-v1",
        "teacher_model": "teacher",
        "prompt_version": "v1",
        "overall": {
            "training_usable": False,
            "recommended_route": "reject",
            "confidence": 0.95,
        },
        "tasks": {"shake": {"probability": 0.9, "confidence": 0.9}},
    }), encoding="utf-8")
    queue = tmp_path / "queue.jsonl"
    queue.write_text(json.dumps({
        "request_id": "one",
        "source_uri": "/readonly/video.mp4",
        "source_class": "supplier_dataset",
        "source_dataset": "supplier-batch-a",
        "supplier_id": "vendor-a",
        "parent_episode_index": 42,
        "split_group": "supplier-batch-a:raw-video:abc",
        "split_group_source": "raw_source_uri",
        "leakage_risk": "low",
        "clip_start_s": 1.0,
        "clip_end_s": 5.0,
        "task": "demo",
        "output_path": str(label),
    }) + "\n", encoding="utf-8")

    summary = build_teacher_training_pool(queue, config, tmp_path / "out")
    row = json.loads((tmp_path / "out/teacher-train-high-confidence.jsonl").read_text())

    assert summary["quality_bands"] == {"strong_positive": 1}
    assert row["distillation"]["split"] == "train"
    assert row["distillation"]["acceptance_authority"] is False
    assert row["vla_pretraining"]["clip_sampler"]["fixed_start_s"] == 1.0
    assert row["source_class"] == "supplier_dataset"
    assert row["source_dataset"] == "supplier-batch-a"
    assert row["supplier_id"] == "vendor-a"
    assert row["parent_episode_index"] == 42
    assert row["distillation"]["split_group"] == "supplier-batch-a:raw-video:abc"
    assert row["distillation"]["split_group_source"] == "raw_source_uri"
    assert row["distillation"]["leakage_risk"] == "low"
