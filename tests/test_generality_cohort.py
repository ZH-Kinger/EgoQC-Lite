from __future__ import annotations

import json
from pathlib import Path

from egoqc.generality_cohort import build_egodex_systems_cohort, plan_generality_cohort


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_generality_cohort_deduplicates_groups_and_holds_out_source(tmp_path: Path) -> None:
    queue = tmp_path / "queue.jsonl"
    rows = []
    for dataset, source_class, groups in (
        ("public-a", "public_dataset", 6),
        ("supplier-a", "supplier_dataset", 8),
    ):
        for group in range(groups):
            for clip in range(2):
                rows.append({
                    "request_id": f"{dataset}-{group}-{clip}",
                    "split_group": f"{dataset}:raw:{group}",
                    "source_dataset": dataset,
                    "source_class": source_class,
                    "selection_source": "bad" if group % 2 else "control",
                })
    _write_jsonl(queue, rows)
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "systems_scale_cohort": {"minimum_unique_clips": 100},
        "model_development": {"target_unique_train_clips": 3},
        "human_gold": {"validation": 2, "in_domain_test": 2, "external_source_test": 4},
    }), encoding="utf-8")

    summary = plan_generality_cohort([queue], protocol, tmp_path / "out", seed=5)

    assert summary["input_rows"] == 28
    assert summary["unique_split_groups"] == 14
    assert summary["dropped_same_group_clips"] == 14
    assert summary["leakage_check_passed"] is True
    assert summary["human_gold_created"] == 0
    external = [json.loads(line) for line in (tmp_path / "out/gold-external-test-candidates.jsonl").read_text().splitlines()]
    assert len(external) == 4
    assert {row["source_class"] for row in external} == {"public_dataset"}
    assert all(row["human_gold_status"] == "candidate_not_labeled" for row in external)


def test_generality_cohort_reports_shortfall_without_reusing_groups(tmp_path: Path) -> None:
    queue = tmp_path / "queue.jsonl"
    _write_jsonl(queue, [{
        "request_id": "one",
        "split_group": "supplier:raw:one",
        "source_dataset": "supplier",
        "source_class": "supplier_dataset",
        "selection_source": "control",
    }])
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "systems_scale_cohort": {"minimum_unique_clips": 10},
        "model_development": {"target_unique_train_clips": 10},
        "human_gold": {"validation": 2, "in_domain_test": 3, "external_source_test": 4},
    }), encoding="utf-8")

    summary = plan_generality_cohort([queue], protocol, tmp_path / "out")

    assert summary["coverage_gaps"]["systems"] == 9
    assert summary["coverage_gaps"]["external_test"] == 4
    assert summary["coverage_gaps"]["validation"] + summary["coverage_gaps"]["in_domain_test"] == 4


def test_generality_cohort_recovers_legacy_oss_dataset_identity(tmp_path: Path) -> None:
    queue = tmp_path / "queue.jsonl"
    _write_jsonl(queue, [{
        "request_id": "legacy",
        "split_group": "oss-unclassified:raw-video:abc",
        "source_dataset": "oss-unclassified",
        "source_class": "supplier_dataset",
        "selection_source": "control",
        "raw_source_uri": "/mnt/data/oss/Task_Family/dataset-123/videos/cam/file.mp4",
    }])
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "systems_scale_cohort": {"minimum_unique_clips": 10},
        "model_development": {"target_unique_train_clips": 0},
        "human_gold": {"validation": 1, "in_domain_test": 0, "external_source_test": 0},
    }), encoding="utf-8")

    summary = plan_generality_cohort([queue], protocol, tmp_path / "out")
    rows = [json.loads(line) for line in (tmp_path / "out/gold-validation-candidates.jsonl").read_text().splitlines()]

    assert summary["systems"]["source_datasets"] == {"oss:dataset-123": 1}
    assert rows[0]["task_family"] == "Task_Family"
    assert rows[0]["source_origin_status"] == "unclassified"
    assert rows[0]["source_class"] == "unclassified_mounted_dataset"


def test_egodex_systems_cohort_is_balanced_and_never_training_eligible(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.jsonl"
    _write_jsonl(inventory, [
        {
            "video_path": f"/mnt/data/egodex/train/{task}/{index}.mp4",
            "task": task,
            "partition": "train",
            "episode_id": f"{task}/{index}",
        }
        for task in ("task-a", "task-b", "task-c")
        for index in range(5)
    ])

    summary = build_egodex_systems_cohort(
        inventory, tmp_path / "out", maximum_videos=9, seed=3
    )
    rows = [json.loads(line) for line in (tmp_path / "out/systems-cohort.jsonl").read_text().splitlines()]

    assert summary["selected_videos"] == 9
    assert summary["tasks"] == 3
    assert summary["minimum_task_videos"] == 3
    assert summary["maximum_task_videos"] == 3
    assert len({row["split_group"] for row in rows}) == 9
    assert not any(row["training_eligible"] for row in rows)
    assert not any(row["accuracy_evaluation_eligible"] for row in rows)
