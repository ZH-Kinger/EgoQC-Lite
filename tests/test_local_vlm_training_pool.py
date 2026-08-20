import json
from pathlib import Path

from egoqc.local_vlm_training_pool import build_local_vlm_training_pool


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_local_pool_is_train_only_and_masks_mano_without_overlay(tmp_path: Path) -> None:
    config = tmp_path / "tasks.json"
    config.write_text(
        json.dumps({"model_tasks": {"shake": {}, "mano_overlay_drift": {}}}),
        encoding="utf-8",
    )
    queue = tmp_path / "queue.jsonl"
    _write_jsonl(
        queue,
        [
            {
                "request_id": "one",
                "source_uri": "/readonly/raw.mp4",
                "source_class": "supplier_dataset",
                "source_dataset": "supplier-a",
                "supplier_id": "vendor-a",
                "split_group": "supplier-a:raw-video:one",
                "split_group_source": "raw_source_uri",
                "leakage_risk": "low",
                "clip_start_s": 1.0,
                "clip_end_s": 5.0,
                "tasks": ["wipe table"],
                "selection_source": "deterministic_bad_frame",
            }
        ],
    )
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions,
        [
            {
                "schema_version": "egoqc-few-b-vlm-benchmark-v1",
                "video_id": "one",
                "structured_json_valid": True,
                "frame_count": 8,
                "maximum_edge": 448,
                "parsed_response": {
                    "f": [["shake", 0.9, 2, 0.1, 0.8, [1, 3]]],
                    "c": 0.95,
                    "a": False,
                },
            }
        ],
    )

    summary = build_local_vlm_training_pool(
        queue, predictions, config, tmp_path / "out"
    )
    row = json.loads((tmp_path / "out/local-teacher-train-high-confidence.jsonl").read_text())

    assert summary["high_confidence_records"] == 1
    assert row["distillation"]["split"] == "train"
    assert row["distillation"]["acceptance_authority"] is False
    assert row["distillation"]["targets"]["shake"] == 0.9
    assert row["distillation"]["label_masks"]["mano_overlay_drift"] == 0
    assert row["distillation"]["label_weights"]["shake"] <= 0.25
    assert row["source_uri"] == "/readonly/raw.mp4"


def test_local_pool_routes_disagreement_and_abstention_to_review(tmp_path: Path) -> None:
    config = tmp_path / "tasks.json"
    config.write_text(json.dumps({"model_tasks": {"shake": {}}}), encoding="utf-8")
    queue = tmp_path / "queue.jsonl"
    _write_jsonl(
        queue,
        [
            {
                "request_id": "bad-clean",
                "source_uri": "/readonly/a.mp4",
                "clip_start_s": 0.0,
                "clip_end_s": 4.0,
                "selection_source": "deterministic_bad_frame",
            },
            {
                "request_id": "abstain",
                "source_uri": "/readonly/b.mp4",
                "clip_start_s": 0.0,
                "clip_end_s": 4.0,
            },
        ],
    )
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions,
        [
            {
                "video_id": "bad-clean",
                "structured_json_valid": True,
                "parsed_response": {"f": [], "c": 0.95, "a": False},
            },
            {
                "video_id": "abstain",
                "structured_json_valid": True,
                "parsed_response": {"f": [], "c": 0.95, "a": True},
            },
        ],
    )

    summary = build_local_vlm_training_pool(
        queue, predictions, config, tmp_path / "out"
    )

    assert summary["records"] == 1
    assert summary["high_confidence_records"] == 0
    assert summary["human_review_records"] == 2
    assert summary["skipped"] == {"model_abstained": 1}
