import hashlib
import json
from pathlib import Path

import av

from create_fixture import create_fixture
from egoqc.gold_review import build_phase_a_review_events


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame_count(path: Path) -> int:
    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(container.streams.video[0]))


def test_phase_a_gold_review_materializes_exact_readonly_episode(tmp_path: Path) -> None:
    dataset = create_fixture(tmp_path / "raw" / "dataset", frames=12, episodes=2)
    source = (
        dataset
        / "videos"
        / "observation.images.ego"
        / "chunk-000"
        / "file-000.mp4"
    )
    before = _sha256(source)
    baseline = tmp_path / "baseline-evidence.jsonl"
    baseline.write_text(
        json.dumps(
            {
                "dataset_id": "fixture:revision",
                "episode_index": 1,
                "length": 12,
                "tasks": ["fixture task"],
                "tier": "bronze",
                "issue_codes": [
                    "bad_frame_ratio_exceeded", "position_jitter", "temporal_spike"
                ],
                "bad_frames": [
                    {"frame_index": 4, "code": "temporal_spike", "side": "left"}
                ],
                "sample_frames": [4],
                "evidence": {"metric:left_position_jitter_p99_mm": 25.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    annotated = tmp_path / "annotated"
    annotated.mkdir()
    annotated_video = annotated / "episode-000001-annotated.mp4"
    annotated_video.write_bytes(b"derived")

    output = tmp_path / "review"
    summary = build_phase_a_review_events(
        dataset,
        baseline,
        output,
        annotated_root=annotated,
    )

    assert summary["events"] == 1
    assert summary["raw_clips"] == 1
    assert summary["annotated_clips"] == 1
    assert _sha256(source) == before
    raw_clip = output / "media" / "episode-000001-raw.mp4"
    assert _frame_count(raw_clip) == 12
    events = json.loads((output / "review-events.json").read_text(encoding="utf-8"))
    event = events[0]
    assert event["kind"] == "episode_qc_gold_review"
    assert event["synthetic"] is False
    assert event["raw_source_readonly"] is True
    assert event["issue_codes"] == ["position_jitter", "temporal_spike"]
    assert event["aggregate_issue_codes"] == ["bad_frame_ratio_exceeded"]
    assert event["mano_overlay_available"] is True


def test_phase_a_gold_review_refuses_output_inside_raw(tmp_path: Path) -> None:
    dataset = create_fixture(tmp_path / "dataset")
    baseline = tmp_path / "baseline.jsonl"
    baseline.write_text("", encoding="utf-8")
    try:
        build_phase_a_review_events(dataset, baseline, dataset / "review")
    except ValueError as error:
        assert "不能位于原始 dataset 内部" in str(error)
    else:
        raise AssertionError("expected readonly boundary failure")
