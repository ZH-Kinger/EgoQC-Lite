import json
from pathlib import Path

from egoqc.few_b_benchmark import parse_structured_response, select_benchmark_rows


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
