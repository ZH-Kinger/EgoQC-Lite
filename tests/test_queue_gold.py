from __future__ import annotations

import json
from pathlib import Path

from egoqc.queue_gold import _dataset_root, build_queue_gold_review


def test_build_queue_gold_review_is_balanced_and_clip_relative(tmp_path: Path) -> None:
    queue = tmp_path / "queue.jsonl"
    rows = []
    for source in ("a", "b"):
        for selection in ("deterministic_bad_frame", "deterministic_low_event_control"):
            for index in range(3):
                rows.append({
                    "request_id": f"{source}-{selection}-{index}",
                    "source_uri": "/readonly/source.mp4",
                    "source_dataset": source,
                    "source_class": "supplier_dataset",
                    "selection_source": selection,
                    "clip_start_s": 10.0,
                    "clip_end_s": 16.0,
                    "clip_start_frame": 300,
                    "event_frames": [330],
                    "event_codes": ["temporal_spike"],
                    "trigger_tasks": ["semantic_camera_shake"],
                    "split_group": f"{source}:video:1",
                })
    queue.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    summary = build_queue_gold_review(
        queue,
        tmp_path / "gold",
        maximum_events=8,
        materialize_media=False,
    )
    events = [
        json.loads(line)
        for line in (tmp_path / "gold/review-events.jsonl").read_text().splitlines()
    ]

    assert summary["review_events"] == 8
    assert summary["source_counts"] == {"a": 4, "b": 4}
    assert summary["selection_counts"] == {
        "deterministic_bad_frame": 4,
        "deterministic_low_event_control": 4,
    }
    assert all(event["review_mode"] == "episode_gold" for event in events)
    assert all(event["sample_frames"] == [30] for event in events)
    assert all(event["raw_source_readonly"] is True for event in events)


def test_dataset_root_is_derived_from_lerobot_video_path() -> None:
    assert _dataset_root(
        "/mnt/data/dataset/videos/observation.images.ego/chunk-000/file-000.mp4"
    ) == Path("/mnt/data/dataset")


def test_build_queue_gold_review_embeds_existing_teacher_prediction(tmp_path: Path) -> None:
    label = tmp_path / "teacher-label.json"
    label.write_text(json.dumps({
        "schema_version": "egoqc-visual-teacher-v1",
        "teacher_model": "teacher",
        "prompt_version": "v1",
        "overall": {
            "training_usable": False,
            "recommended_route": "reject",
            "confidence": 0.95,
        },
        "tasks": {
            "hand_absent": {"probability": 0.8, "confidence": 0.9},
            "severe_occlusion": {"probability": 0.2, "confidence": 0.8},
        },
        "findings": [{
            "category": "hand_absent",
            "start_s": 0.5,
            "end_s": 1.0,
        }],
        "summary": "hand leaves the frame",
    }) + "\n", encoding="utf-8")
    queue = tmp_path / "queue.jsonl"
    queue.write_text(json.dumps({
        "request_id": "public-one",
        "source_uri": "/readonly/public.mp4",
        "source_dataset": "public",
        "source_class": "public_dataset",
        "selection_source": "public_review",
        "clip_start_s": 10.0,
        "clip_end_s": 16.0,
        "split_group": "public:raw-video:one",
        "output_path": str(label),
    }) + "\n", encoding="utf-8")

    build_queue_gold_review(queue, tmp_path / "gold", maximum_events=1)
    event = json.loads((tmp_path / "gold/review-events.jsonl").read_text())

    assert event["issue_codes"] == ["hand_absent"]
    assert event["issue_labels"]["hand_absent"] == "手离画超过标准"
    assert event["baseline_tier"] == "bronze"
    assert event["sample_frames"] == [15, 30]
    assert event["machine_assessment_source"] == "api_vlm_teacher"
    assert event["teacher_preview"]["summary"] == "hand leaves the frame"
    assert event["unavailable_gold_labels"] == ["mano_overlay_drift"]
    assert "mano_overlay_drift" not in {
        row["code"] for row in event["gold_labels"]
    }
