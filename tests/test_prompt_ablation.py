import json
from pathlib import Path

import pytest

from egoqc.prompt_ablation import compare_paired_prompt_predictions


def _write_root(root: Path, rows: list[dict]) -> None:
    root.mkdir()
    (root / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _row(identity: str, findings: list, confidence: float = 1.0) -> dict:
    return {
        "record_id": identity,
        "selection_source": "bad" if identity == "a" else "control",
        "structured_json_valid": True,
        "parsed_response": {"f": findings, "c": confidence, "a": False},
    }


def test_compare_paired_prompt_predictions_uses_candidate_subset(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_root(
        baseline,
        [_row("extra", []), _row("a", [["blur", 0.9]]), _row("b", [])],
    )
    _write_root(
        candidate,
        [_row("a", []), _row("b", [["shake", 0.8]], confidence=0.8)],
    )

    summary = compare_paired_prompt_predictions(
        baseline, candidate, tmp_path / "out"
    )
    report = json.loads((tmp_path / "out" / "comparison.json").read_text())

    assert summary["paired_clips"] == 2
    assert report["baseline"]["task_trigger_counts"] == {"blur": 1}
    assert report["candidate"]["task_trigger_counts"] == {"shake": 1}
    assert report["paired_changes"]["changed_task_set_rate"] == 1.0
    assert report["formal_accuracy_measured"] is False


def test_compare_paired_prompt_predictions_rejects_missing_id(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_root(baseline, [_row("a", [])])
    _write_root(candidate, [_row("b", [])])

    with pytest.raises(ValueError, match="missing 1 candidate ids"):
        compare_paired_prompt_predictions(baseline, candidate, tmp_path / "out")
