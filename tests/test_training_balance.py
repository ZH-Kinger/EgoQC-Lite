from __future__ import annotations

import json
from pathlib import Path

from egoqc.training_balance import build_training_sampling_weights


def test_training_weights_upweight_rare_semantics_and_groups(tmp_path: Path) -> None:
    source = tmp_path / "records.jsonl"
    rows = []
    for index in range(8):
        rows.append({
            "request_id": f"common-{index}",
            "split_group": "group-common",
            "task_taxonomy": {
                "normalized_task": "common task",
                "interaction_primitives": ["pick_place"],
            },
        })
    rows.append({
        "request_id": "rare",
        "split_group": "group-rare",
        "task_taxonomy": {
            "normalized_task": "rare task",
            "interaction_primitives": ["zip_unzip"],
        },
    })
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))

    summary = build_training_sampling_weights(source, tmp_path / "out")
    weighted = [
        json.loads(line)
        for line in (tmp_path / "out/weighted-training-candidates.jsonl").read_text().splitlines()
    ]
    weights = {row["request_id"]: row["training_sample_weight"] for row in weighted}

    assert summary["records"] == 9
    assert summary["interaction_primitives"] == 2
    assert weights["rare"] > weights["common-0"]
    assert min(weights.values()) >= 0.25
    assert max(weights.values()) <= 4.0
