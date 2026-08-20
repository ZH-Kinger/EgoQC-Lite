import json
from pathlib import Path

from egoqc.few_b_benchmark import (
    _clip_window,
    freeze_few_b_samples,
    normalize_sparse_findings,
    parse_structured_response,
    select_benchmark_rows,
)


def test_parse_structured_response_accepts_plain_and_fenced_json() -> None:
    expected = {"confidence": 0.8, "abstain": False}
    assert parse_structured_response(json.dumps(expected))[0] == expected
    assert parse_structured_response("```json\n" + json.dumps(expected) + "\n```")[0] == expected
    parsed, error = parse_structured_response("not-json")
    assert parsed is None
    assert error


def test_select_benchmark_rows_is_deterministic_and_requires_local_media(tmp_path: Path) -> None:
    paths = []
    for index in range(4):
        path = tmp_path / f"{index}.mp4"
        path.write_bytes(b"video")
        paths.append(path)
    rows = [
        {"video_id": f"v{index}", "source_uri": str(path)}
        for index, path in enumerate(paths)
    ] + [{"video_id": "missing", "source_uri": str(tmp_path / "missing.mp4")}]
    first = select_benchmark_rows(rows, 2, 17)
    second = select_benchmark_rows(list(reversed(rows)), 2, 17)
    assert [row["video_id"] for row in first] == [row["video_id"] for row in second]
    assert len(first) == 2


def test_normalize_sparse_findings_maps_indices_and_preserves_codes() -> None:
    normalized, error = normalize_sparse_findings(
        {"f": [["1", 0.8, 2, 0.1, 0.4, [2]], ["first", 0.5, 1, 0, 1, [0]]]},
        ["first", "second"],
    )
    assert error is None
    assert normalized["f"][0][0] == "second"
    assert normalized["f"][1][0] == "first"


def test_balanced_weak_selection_uses_positive_and_negative_rows(tmp_path: Path) -> None:
    rows = []
    for index in range(8):
        path = tmp_path / f"balanced-{index}.mp4"
        path.write_bytes(b"video")
        rows.append(
            {
                "video_id": f"v{index}",
                "source_uri": str(path),
                "distillation": {"targets": {"issue": 0.9 if index < 4 else 0.0}},
            }
        )
    selected = select_benchmark_rows(rows, 6, 17, strategy="balanced_weak")
    positives = sum(
        row["distillation"]["targets"]["issue"] >= 0.5 for row in selected
    )
    assert positives == 3


def test_clip_window_accepts_teacher_queue_fields() -> None:
    assert _clip_window(
        {"clip_start_s": 19.3, "clip_end_s": 25.9, "duration_s": 42.8}
    ) == (19.3, 25.9)


def test_freeze_few_b_samples_writes_balanced_hashed_manifest(tmp_path: Path) -> None:
    rows = []
    for index in range(8):
        video = tmp_path / f"freeze-{index}.mp4"
        video.write_bytes(b"video")
        rows.append(
            {
                "video_id": f"v{index}",
                "source_uri": str(video),
                "source_dataset": "public",
                "distillation": {
                    "split_group": f"group-{index}",
                    "targets": {"issue": 0.9 if index < 4 else 0.0},
                },
            }
        )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    summary = freeze_few_b_samples(
        manifest,
        tmp_path / "frozen",
        maximum_clips=6,
        seed=31,
        selection_strategy="balanced_weak",
    )
    assert summary["samples"] == 6
    assert summary["positive_rows_by_weak_teacher"] == 3
    assert summary["negative_rows_by_weak_teacher"] == 3
    assert summary["split_groups"] == 6
    assert len(summary["samples_sha256"]) == 64
