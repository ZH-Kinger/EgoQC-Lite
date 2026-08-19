from __future__ import annotations

import json
from pathlib import Path

import pytest

from egoqc.training_partition import freeze_training_partition


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_freeze_training_partition_quarantines_complete_raw_video_groups(tmp_path: Path) -> None:
    queue = tmp_path / "teacher.jsonl"
    teacher_rows = []
    for group in range(8):
        for clip in range(2):
            teacher_rows.append({
                "request_id": f"request-{group}-{clip}",
                "split_group": f"source:raw-video:{group}",
                "source_dataset": "source-a" if group % 2 == 0 else "source-b",
                "selection_source": "bad" if clip == 0 else "control",
            })
    _write_jsonl(queue, teacher_rows)

    gold = tmp_path / "gold.jsonl"
    gold_rows = [
        {
            "event_id": f"event-{group}",
            "split_group": f"source:raw-video:{group}",
            "source_dataset": "source-a" if group % 2 == 0 else "source-b",
            "selection_source": "bad" if group % 2 == 0 else "control",
        }
        for group in range(4)
    ]
    _write_jsonl(gold, gold_rows)

    summary = freeze_training_partition(queue, gold, tmp_path / "out", seed=7)
    train = [json.loads(line) for line in (tmp_path / "out/teacher-train-queue.jsonl").read_text().splitlines()]
    quarantine = [json.loads(line) for line in (tmp_path / "out/quarantined-teacher-requests.jsonl").read_text().splitlines()]
    validation = [json.loads(line) for line in (tmp_path / "out/gold-validation-events.jsonl").read_text().splitlines()]
    test = [json.loads(line) for line in (tmp_path / "out/gold-test-events.jsonl").read_text().splitlines()]

    assert len(train) == 8
    assert len(quarantine) == 8
    assert {row["split_group"] for row in train}.isdisjoint(
        {row["split_group"] for row in gold_rows}
    )
    assert {row["split_group"] for row in validation}.isdisjoint(
        {row["split_group"] for row in test}
    )
    assert validation and test
    assert all(row["dataset_role"] == "teacher_train_candidate" for row in train)
    assert all(row["quarantine_reason"] for row in quarantine)
    assert summary["leakage_check_passed"] is True
    assert not any(summary["split_group_intersections"].values())


def test_freeze_training_partition_requires_split_groups(tmp_path: Path) -> None:
    queue = tmp_path / "teacher.jsonl"
    gold = tmp_path / "gold.jsonl"
    _write_jsonl(queue, [{"request_id": "one"}])
    _write_jsonl(gold, [
        {"event_id": "gold-one", "split_group": "g1"},
        {"event_id": "gold-two", "split_group": "g2"},
    ])

    with pytest.raises(ValueError, match="split_group"):
        freeze_training_partition(queue, gold, tmp_path / "out")
