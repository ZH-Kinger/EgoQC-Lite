import json
from pathlib import Path

from egoqc.model_ablation import plan_model_ablation


ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_plan_model_ablation_freezes_few_b_matrix_and_blocks_small_gold(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    test = tmp_path / "test.jsonl"
    _write_jsonl(train, [{"video_id": "train-1"}, {"video_id": "train-2"}])
    _write_jsonl(validation, [{"video_id": "val-1", "decision": "pass"}])
    _write_jsonl(test, [{"video_id": "test-1", "decision": "fail"}])

    output = tmp_path / "plan"
    summary = plan_model_ablation(
        train,
        validation,
        test,
        ROOT / "config" / "qc_student_deployment_v1.json",
        output,
    )

    assert summary["runs"] == 19
    assert summary["benchmarks"] == 6
    assert summary["accuracy_claim_authorized"] is False
    plan = json.loads((output / "experiment-plan.json").read_text(encoding="utf-8"))
    assert plan["fair_comparison"]["same_frame_count"] == 8
    assert plan["data_guards"]["split_fingerprints_distinct"] is True
    runs = [json.loads(line) for line in (output / "run-matrix.jsonl").read_text().splitlines()]
    assert {run["system_id"] for run in runs} >= {
        "qwen3_vl_2b_full_sft",
        "qwen3_vl_4b_full_sft",
        "qwen3_vl_8b_full_sft",
        "scout_to_qwen3_vl_4b",
    }


def test_plan_model_ablation_rejects_duplicate_seeds(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [{"video_id": "a", "decision": "pass"}])
    try:
        plan_model_ablation(
            source,
            source,
            source,
            ROOT / "config" / "qc_student_deployment_v1.json",
            tmp_path / "plan",
            seeds=(17, 17),
        )
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate seeds should fail")
