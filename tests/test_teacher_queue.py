from __future__ import annotations

import json
from pathlib import Path

from egoqc.teacher_queue import merge_teacher_queues


def _write_queue(path: Path, source: str, selection: str, count: int) -> None:
    rows = [
        {
            "request_id": f"{source}-{selection}-{index}",
            "source_dataset": source,
            "source_class": "supplier_dataset",
            "selection_source": selection,
            "split_group": f"{source}:video:{index // 2}",
            "trigger_tasks": ["semantic_camera_shake"] if selection == "positive" else [],
        }
        for index in range(count)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_merge_teacher_queues_round_robins_source_and_selection(tmp_path: Path) -> None:
    queues = []
    for source in ("source-a", "source-b"):
        for selection in ("positive", "control"):
            path = tmp_path / f"{source}-{selection}.jsonl"
            _write_queue(path, source, selection, 4)
            queues.append(path)

    summary = merge_teacher_queues(
        queues,
        tmp_path / "merged",
        maximum_requests=8,
        seed=9,
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "merged/teacher-api-queue.jsonl").read_text().splitlines()
    ]

    assert summary["selected_requests"] == 8
    assert summary["source_counts"] == {"source-a": 4, "source-b": 4}
    assert summary["selection_counts"] == {"control": 4, "positive": 4}
    assert summary["external_transfer"]["requires_explicit_runtime_authorization"] is True
    assert len({row["request_id"] for row in rows}) == 8
