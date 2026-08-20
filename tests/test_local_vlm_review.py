import json
from pathlib import Path

from egoqc.local_vlm_review import prepare_local_vlm_review_queue


def test_prepare_local_vlm_review_queue_is_non_authoritative(tmp_path: Path) -> None:
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        json.dumps(
            {
                "request_id": "clip-a",
                "source_uri": "/readonly/a.mp4",
                "source_class": "supplier_dataset",
                "clip_start_s": 10.0,
                "clip_end_s": 14.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    root = tmp_path / "benchmark"
    root.mkdir()
    (root / "benchmark.json").write_text(
        json.dumps(
            {
                "model_id": "local-8b",
                "input_protocol": {
                    "prompt_version": "v2",
                    "task_order": ["hand_absent", "task_label_mismatch"],
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "predictions.jsonl").write_text(
        json.dumps(
            {
                "record_id": "clip-a",
                "structured_json_valid": True,
                "parsed_response": {
                    "f": [["hand_absent", 0.9, 2, 0.25, 0.75, [1, 2]]],
                    "c": 0.95,
                    "a": False,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = prepare_local_vlm_review_queue(queue, root, tmp_path / "out")
    review = json.loads((tmp_path / "out/review-queue.jsonl").read_text())
    label = json.loads(Path(review["output_path"]).read_text())

    assert summary["review_requests"] == 1
    assert summary["machine_suggestions_are_gold"] is False
    assert label["acceptance_authority"] is False
    assert label["training_label_authority"] is False
    assert label["findings"][0]["start_s"] == 1.0
    assert label["findings"][0]["end_s"] == 3.0
