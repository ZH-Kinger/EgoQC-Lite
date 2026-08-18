import json
from pathlib import Path

from tools.analyze_teacher_labels import analyze


def test_analyze_teacher_labels(tmp_path: Path) -> None:
    label_path = tmp_path / "label.json"
    label_path.write_text(
        json.dumps(
            {
                "overall": {
                    "training_usable": True,
                    "recommended_route": "accept",
                    "confidence": 0.9,
                    "allowed_uses": ["video_pretraining"],
                },
                "tasks": {"semantic_camera_shake": {"probability": 0.8}},
                "findings": [{"category": "blur", "severity": "warning"}],
                "missing_annotations": ["mano_parameters"],
            }
        ),
        encoding="utf-8",
    )
    queue_path = tmp_path / "queue.jsonl"
    queue_path.write_text(
        json.dumps(
            {
                "request_id": "one",
                "candidate_tier": "hard-negative",
                "output_path": str(label_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = analyze(queue_path)

    assert summary["labeled"] == 1
    assert summary["training_usable"] == 1
    assert summary["selection_teacher_conflicts"]["hard_negative_accepted"] == 1
    assert summary["positive_tasks_probability_ge_0_8"]["semantic_camera_shake"] == 1
    assert summary["top_finding_categories"]["blur"] == 1
