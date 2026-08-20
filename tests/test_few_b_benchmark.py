import json
from pathlib import Path

import av
import numpy as np

from egoqc.few_b_benchmark import (
    _activity_text,
    _prompt,
    _clip_window,
    _load_resumable_results,
    _prefetched_decodes,
    load_frame_cache_index,
    _sample_video_frames,
    freeze_few_b_samples,
    normalize_sparse_findings,
    parse_structured_response,
    select_benchmark_rows,
)


def test_prompt_reads_lerobot_tasks_and_guards_missing_overlay() -> None:
    task_config = json.loads(Path("config/visual_model_tasks.json").read_text())
    activity = _activity_text({"tasks": ["把杯子放到桌上"]})
    prompt = _prompt(task_config, activity, 8, overlay_available=False)

    assert activity == "把杯子放到桌上"
    assert "把杯子放到桌上" in prompt
    assert "不得输出mano_overlay_drift" in prompt
    assert "不可见信号由外部规则判断" in prompt


def _write_video(path: Path, frames: int = 180) -> None:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"
        for index in range(frames):
            array = np.full((48, 64, 3), index % 255, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


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


def test_sample_video_frames_decodes_one_forward_window(tmp_path: Path) -> None:
    video = tmp_path / "sample.mp4"
    _write_video(video)

    frames = _sample_video_frames(video, 1.0, 5.0, 8)

    assert len(frames) == 8
    means = [float(np.asarray(frame).mean()) for frame in frames]
    assert means == sorted(means)


def test_resume_only_accepts_same_protocol_and_selected_ids(tmp_path: Path) -> None:
    output = tmp_path / "out"
    output.mkdir()
    rows = [{"request_id": "keep"}, {"request_id": "pending"}]
    (output / "predictions.partial.jsonl").write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in [
                {"video_id": "keep", "frame_count": 8, "maximum_edge": 448},
                {"video_id": "wrong-edge", "frame_count": 8, "maximum_edge": 640},
                {"video_id": "not-selected", "frame_count": 8, "maximum_edge": 448},
            ]
        ),
        encoding="utf-8",
    )

    resumed = _load_resumable_results(output, rows, frame_count=8, maximum_edge=448)

    assert [row["video_id"] for row in resumed] == ["keep"]


def test_prefetched_decodes_preserve_row_order(tmp_path: Path) -> None:
    rows = []
    for index in range(2):
        video = tmp_path / f"prefetch-{index}.mp4"
        _write_video(video, frames=180)
        rows.append(
            {
                "request_id": f"row-{index}",
                "source_uri": str(video),
                "clip_start_s": 1.0,
                "clip_end_s": 3.0,
                "duration_s": 6.0,
            }
        )

    decoded = list(_prefetched_decodes(rows, frame_count=4, workers=2))

    assert [item[0]["request_id"] for item in decoded] == ["row-0", "row-1"]
    assert all(len(item[4]) == 4 for item in decoded)


def test_decode_uses_valid_frame_cache_without_raw_video_decode(tmp_path: Path) -> None:
    from egoqc.few_b_frame_cache import predecode_few_b_frame_cache

    video = tmp_path / "cache-source.mp4"
    _write_video(video, frames=180)
    row = {
        "request_id": "cached-row",
        "source_uri": str(video),
        "clip_start_s": 1.0,
        "clip_end_s": 3.0,
        "duration_s": 6.0,
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    cache = tmp_path / "cache"
    predecode_few_b_frame_cache(
        manifest,
        cache,
        maximum_clips=1,
        frame_count=4,
        maximum_edge=64,
        workers=1,
    )

    decoded = list(
        _prefetched_decodes(
            [row],
            frame_count=4,
            workers=1,
            maximum_edge=64,
            frame_cache_root=cache,
            frame_cache_index=load_frame_cache_index(cache),
            require_frame_cache=True,
        )
    )

    assert decoded[0][6] == "frame_cache"
    assert len(decoded[0][4]) == 4


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
