import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from create_fixture import create_fixture
from egoqc.interventions import (
    apply_intervention,
    plan_qc_interventions,
    run_qc_interventions,
)


ROOT = Path(__file__).parents[1]
INTERVENTION_CONFIG = ROOT / "config" / "qc_interventions_phase_a.json"
QC_CONFIG = ROOT / "config" / "default.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_plan_and_run_interventions_without_mutating_source(tmp_path: Path) -> None:
    dataset = create_fixture(tmp_path / "raw" / "dataset", frames=20, episodes=2)
    source = dataset / "data" / "chunk-000" / "file-000.parquet"
    before = _sha256(source)

    plan = plan_qc_interventions(
        dataset,
        INTERVENTION_CONFIG,
        tmp_path / "derived" / "plan",
        maximum_episodes=1,
        seed=7,
    )
    assert plan["interventions"] == 14
    assert plan["materialized_source_copies"] == 0
    manifest_rows = _jsonl(tmp_path / "derived" / "plan" / "interventions.jsonl")
    assert all(row["view"]["materialized"] is False for row in manifest_rows)
    assert all("/raw/" not in json.dumps(row) for row in manifest_rows)

    summary = run_qc_interventions(
        dataset,
        tmp_path / "derived" / "plan" / "interventions.jsonl",
        QC_CONFIG,
        tmp_path / "derived" / "run",
    )
    assert summary["interventions"] == 14
    assert summary["source_shards_read"] == 1
    assert summary["materialized_source_copies"] == 0
    assert summary["by_family"]["timestamp_offset"]["target_hit_rate"] == 1.0
    assert summary["by_family"]["wrist_position_offset"]["target_hit_rate"] == 1.0
    assert summary["by_family"]["state_mask_dropout"]["target_hit_rate"] == 1.0
    assert (tmp_path / "derived" / "run" / "evidence-deltas.jsonl").is_file()
    assert _sha256(source) == before


def test_apply_intervention_is_an_immutable_arrow_view(tmp_path: Path) -> None:
    dataset = create_fixture(tmp_path / "dataset")
    source = dataset / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(source)
    original = table["timestamp"].to_pylist()
    changed = apply_intervention(
        table,
        "timestamp_offset",
        {"offset_s": 0.025},
        3,
        8,
    )
    assert table["timestamp"].to_pylist() == original
    assert changed["timestamp"].to_pylist() != original


def test_intervention_outputs_are_rejected_inside_raw_dataset(tmp_path: Path) -> None:
    dataset = create_fixture(tmp_path / "dataset")
    with pytest.raises(ValueError, match="不能位于原始 dataset 内部"):
        plan_qc_interventions(
            dataset,
            INTERVENTION_CONFIG,
            dataset / "derived-interventions",
        )


def test_run_detects_source_drift_after_planning(tmp_path: Path) -> None:
    dataset = create_fixture(tmp_path / "raw" / "dataset")
    plan_qc_interventions(
        dataset,
        INTERVENTION_CONFIG,
        tmp_path / "derived" / "plan",
        maximum_episodes=1,
    )
    source = dataset / "data" / "chunk-000" / "file-000.parquet"
    source.touch()
    with pytest.raises(RuntimeError, match="plan 后发生变化"):
        run_qc_interventions(
            dataset,
            tmp_path / "derived" / "plan" / "interventions.jsonl",
            QC_CONFIG,
            tmp_path / "derived" / "run",
        )
