import json
from pathlib import Path

from egoqc.few_b_comparison import compare_few_b_benchmarks


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _jsonl(path: Path, values: list[dict]) -> None:
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


def _benchmark(root: Path, model: str, probability: float) -> None:
    root.mkdir()
    protocol = {
        "frame_count": 8,
        "maximum_edge": 448,
        "selection_seed": 31,
        "selection_strategy": "balanced_weak",
        "maximum_clips": 1,
        "max_new_tokens": 128,
        "wire_output_schema": "compact_sparse_findings_v1",
    }
    _json(
        root / "benchmark.json",
        {
            "model_id": model,
            "parameter_count": 2_000_000_000,
            "model_artifact": {"bytes": 4_000_000_000},
            "model_memory_mb": 4096,
            "latency_seconds": {"total_p50": 1.0, "total_p95": 1.2},
            "video_hours_per_wall_hour": 5.0,
            "input_protocol": protocol,
        },
    )
    _jsonl(
        root / "predictions.jsonl",
        [{"video_id": "v1", "structured_json_valid": True, "parsed_response": {"f": [["issue", probability, 2, 0, 1, [0]]]}, "output_tokens": 20, "peak_inference_vram_mb": 4300}],
    )


def test_compare_few_b_benchmarks_writes_paper_table(tmp_path: Path) -> None:
    teacher = tmp_path / "teacher.jsonl"
    _jsonl(teacher, [{"video_id": "v1", "distillation": {"targets": {"issue": 0.9}}}])
    first = tmp_path / "first"
    second = tmp_path / "second"
    _benchmark(first, "2b", 0.8)
    _benchmark(second, "4b", 0.2)
    output = tmp_path / "comparison"
    summary = compare_few_b_benchmarks([first, second], teacher, output)
    assert summary["systems"] == 2
    assert summary["formal_accuracy_measured"] is False
    comparison = json.loads((output / "comparison.json").read_text())
    assert comparison["systems"][0]["weak_teacher_agreement"]["recall"] == 1.0
    assert (output / "comparison.tsv").is_file()
    assert "not human accuracy" in (output / "comparison.md").read_text()
