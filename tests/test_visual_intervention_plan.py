from __future__ import annotations

import json
from pathlib import Path

from egoqc.visual_intervention_plan import build_visual_intervention_plan


def test_visual_interventions_are_balanced_lazy_and_train_only(tmp_path: Path) -> None:
    records = tmp_path / "records.jsonl"
    rows = [
        {
            "request_id": f"r-{index}",
            "split_group": f"g-{index // 2}",
            "source_dataset": "source-a" if index % 2 else "source-b",
            "source_uri": f"/raw/{index}.mp4",
            "training_sample_weight": 1.0,
            "task_taxonomy": {
                "normalized_task": f"task {index}",
                "interaction_primitives": ["pick_place" if index < 8 else "zip_unzip"],
            },
        }
        for index in range(12)
    ]
    records.write_text("".join(json.dumps(row) + "\n" for row in rows))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "families": {
            "gaussian_blur": {"required_capability": "raw_rgb", "target_tasks": ["unusable_visual_quality"], "parameters": {"sigma": 3}},
            "task_text_swap": {"required_capability": "task_text", "target_tasks": ["task_label_mismatch"], "parameters": {}},
            "overlay_translation": {"required_capability": "mano_overlay", "target_tasks": ["mano_overlay_drift"], "parameters": {}},
        },
        "real_only_tasks": ["persistent_extra_hands"],
    }))

    summary = build_visual_intervention_plan(records, config, tmp_path / "out", maximum_records=10)
    plans = [json.loads(line) for line in (tmp_path / "out/visual-interventions.jsonl").read_text().splitlines()]

    assert summary["interventions"] == 10
    assert summary["materialized_images"] == 0
    assert set(summary["counts_by_family"]) == {"gaussian_blur", "task_text_swap"}
    assert summary["unavailable_families"] == ["overlay_translation"]
    assert all(row["dataset_role"] == "synthetic_train_only" for row in plans)
    assert all(row["synthetic"] and not row["gold"] for row in plans)
    assert all(row["materialization"] == "lazy_on_cached_frames" for row in plans)
    swaps = [row for row in plans if row["family"] == "task_text_swap"]
    assert swaps and all(row["donor_request_id"] for row in swaps)
