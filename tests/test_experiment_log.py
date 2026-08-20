import json
from pathlib import Path


def test_experiment_run_log_is_complete_unique_and_sanitized() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "artifacts/experiments/experiment-run-log.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    assert len(rows) >= 12
    assert len({row["experiment_id"] for row in rows}) == len(rows)
    for row in rows:
        assert row["status"]
        assert row["objective"]
        assert row["claim_boundary"]
        assert row["raw_source_readonly"] is True
        serialized = json.dumps(row, ensure_ascii=False)
        assert "/mnt/data/" not in serialized
        assert "121.43." not in serialized


def test_logged_evidence_paths_exist_in_repository() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "artifacts/experiments/experiment-run-log.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    for row in rows:
        for evidence in row.get("evidence", []):
            assert (root / evidence).is_file(), evidence
