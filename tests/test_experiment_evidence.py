from __future__ import annotations

from egoqc.experiment_evidence import select_evidence_events


def _row(index: int, selection: str, ratio: float) -> dict:
    return {
        "event_id": f"event-{index}",
        "clip": "/derived/raw.mp4",
        "annotated_clip_path": "/derived/overlay.mp4",
        "selection_source": selection,
        "metrics": {"bad_frame_ratio": ratio},
    }


def test_evidence_selection_keeps_contrast_strata_and_highest_risk() -> None:
    rows = [
        _row(1, "deterministic_bad_frame", 0.1),
        _row(2, "deterministic_bad_frame", 0.4),
        _row(3, "deterministic_bad_frame", 0.3),
        _row(4, "deterministic_clean_gap_control", 0.02),
        _row(5, "deterministic_low_event_control", 0.08),
    ]
    selected = select_evidence_events(
        rows,
        maximum_rule_positive=2,
        maximum_clean_control=1,
        maximum_low_event_control=1,
        seed=7,
    )

    assert len(selected) == 4
    positives = [row for row in selected if row["evidence_bucket"] == "rule-positive-high"]
    assert [row["event_id"] for row in positives] == ["event-2", "event-3"]
    assert {row["evidence_bucket"] for row in selected} == {
        "rule-positive-high", "clean-control", "low-event-control"
    }


def test_evidence_selection_does_not_call_missing_media_rows() -> None:
    selected = select_evidence_events([
        {"event_id": "missing", "selection_source": "deterministic_bad_frame"}
    ])
    assert selected == []
