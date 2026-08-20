from __future__ import annotations

import json
from pathlib import Path

from egoqc.teacher_queue import (
    build_overlay_teacher_queue,
    merge_teacher_queues,
    normalize_teacher_queue_provenance,
)


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


def test_merge_teacher_queues_can_keep_one_clip_per_split_group(tmp_path: Path) -> None:
    queue = tmp_path / "queue.jsonl"
    _write_queue(queue, "source-a", "positive", 8)
    summary = merge_teacher_queues(
        [queue],
        tmp_path / "grouped",
        maximum_requests=8,
        seed=9,
        one_per_split_group=True,
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "grouped/teacher-api-queue.jsonl").read_text().splitlines()
    ]
    assert summary["input_unique_requests"] == 8
    assert summary["selected_requests"] == 4
    assert summary["dropped_split_group_duplicates"] == 4
    assert len({row["split_group"] for row in rows}) == len(rows)


def test_build_overlay_teacher_queue_preserves_raw_provenance(tmp_path: Path) -> None:
    overlay = tmp_path / "overlay.mp4"
    overlay.write_bytes(b"derived")
    queue = tmp_path / "queue.jsonl"
    queue.write_text(json.dumps({
        "request_id": "one",
        "source_uri": "/readonly/raw.mp4",
        "source_class": "supplier_dataset",
        "source_dataset": "supplier-a",
        "clip_start_s": 10.0,
        "clip_end_s": 16.0,
        "trigger_tasks": ["semantic_camera_shake"],
        "output_path": "/old/label.json",
    }) + "\n", encoding="utf-8")
    events = tmp_path / "events.jsonl"
    events.write_text(json.dumps({
        "video_id": "one",
        "annotated_clip_path": str(overlay),
    }) + "\n", encoding="utf-8")

    summary = build_overlay_teacher_queue(queue, events, tmp_path / "out")
    row = json.loads((tmp_path / "out/teacher-api-queue.jsonl").read_text())

    assert summary["requests"] == 1
    assert row["source_uri"] == str(overlay.resolve())
    assert row["raw_source_uri"] == "/readonly/raw.mp4"
    assert row["clip_start_s"] == 0.0
    assert row["clip_end_s"] == 6.0
    assert row["source_clip_start_s"] == 10.0
    assert row["visual_evidence"] == "mano_mesh_skeleton_overlay"
    assert "mano_overlay_drift" in row["trigger_tasks"]
    assert row["output_path"].startswith(str((tmp_path / "out").resolve()))


def test_normalize_legacy_egodex_queue_adds_safe_split_provenance(tmp_path: Path) -> None:
    queue = tmp_path / "queue.jsonl"
    queue.write_text(json.dumps({
        "request_id": "egodex-one",
        "source_uri": "/readonly/egodex/fold_paper/1.mp4",
        "task": "fold_paper",
        "selection_source": "egodex_review",
    }) + "\n", encoding="utf-8")

    summary = normalize_teacher_queue_provenance(queue, tmp_path / "out")
    row = json.loads((tmp_path / "out/teacher-api-queue.jsonl").read_text())

    assert summary["requests"] == 1
    assert summary["split_groups"] == 1
    assert row["source_class"] == "public_dataset"
    assert row["source_dataset"] == "egodex"
    assert row["supplier_id"] == "public:egodex"
    assert row["tasks"] == ["fold_paper"]
    assert row["split_group"].startswith("egodex:raw-video:")
    assert row["split_group_source"] == "raw_source_uri"
    assert row["raw_source_readonly"] is True
