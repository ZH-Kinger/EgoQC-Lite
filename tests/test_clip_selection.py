from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from create_fixture import create_fixture
from egoqc.clip_selection import plan_qc_clips


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class ClipSelectionTest(unittest.TestCase):
    def test_clip_ids_are_unique_across_single_episode_dataset_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task_config = Path(__file__).parents[1] / "config" / "visual_model_tasks.json"
            clip_ids = []
            for name in ("dataset-a", "dataset-b"):
                dataset = create_fixture(root / name, frames=300)
                quality = root / f"{name}-quality"
                quality.mkdir()
                (quality / "episodes.jsonl").write_text(
                    json.dumps({"episode_index": 0, "length": 300}) + "\n",
                    encoding="utf-8",
                )
                (quality / "bad_frames.jsonl").write_text(
                    json.dumps({
                        "episode_index": 0,
                        "frame_index": 150,
                        "code": "temporal_spike",
                        "severity": "error",
                    }) + "\n",
                    encoding="utf-8",
                )
                plan_qc_clips(
                    dataset,
                    quality,
                    root / f"{name}-out",
                    task_config,
                    maximum_clips=1,
                    minimum_control_clips=0,
                    source_class="supplier_dataset",
                    source_dataset="shared-source",
                    supplier_id="shared-supplier",
                )
                queue = _read_jsonl(root / f"{name}-out" / "teacher-api-queue.jsonl")
                clip_ids.append(queue[0]["request_id"])

            self.assertEqual(len(set(clip_ids)), 2)

    def test_merges_events_keeps_aggregate_offset_and_skips_rule_only_api(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = create_fixture(root / "dataset", frames=300, episodes=2)
            quality = root / "quality"
            quality.mkdir()
            episodes = [
                {"episode_index": 0, "length": 300},
                {"episode_index": 1, "length": 300},
            ]
            events = [
                {
                    "episode_index": 1,
                    "frame_index": 30,
                    "code": "temporal_spike",
                    "severity": "error",
                },
                {
                    "episode_index": 1,
                    "frame_index": 45,
                    "code": "mask_flicker",
                    "severity": "warning",
                },
                {
                    "episode_index": 1,
                    "frame_index": 240,
                    "code": "timestamp_mismatch",
                    "severity": "error",
                },
            ]
            (quality / "episodes.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in episodes),
                encoding="utf-8",
            )
            (quality / "bad_frames.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in events),
                encoding="utf-8",
            )
            task_config = Path(__file__).parents[1] / "config" / "visual_model_tasks.json"
            output = root / "clips"

            summary = plan_qc_clips(
                dataset,
                quality,
                output,
                task_config,
                minimum_control_clips=1,
                seed=3,
            )

            clips = _read_jsonl(output / "clip-candidates.jsonl")
            queue = _read_jsonl(output / "teacher-api-queue.jsonl")
            visual = next(row for row in clips if "temporal_spike" in row["event_codes"])
            rule_only = next(row for row in clips if "timestamp_mismatch" in row["event_codes"])

            self.assertEqual(visual["event_frames"], [30, 45])
            self.assertGreaterEqual(visual["clip_end_s"] - visual["clip_start_s"], 4.0)
            self.assertLessEqual(visual["clip_end_s"] - visual["clip_start_s"], 8.0)
            self.assertGreaterEqual(visual["clip_start_s"], 10.0)
            self.assertGreaterEqual(visual["duration_s"], visual["clip_end_s"])
            self.assertNotIn(rule_only["clip_id"], {row["request_id"] for row in queue})
            self.assertIn(visual["clip_id"], {row["request_id"] for row in queue})
            self.assertEqual(summary["teacher_api_requests"], len(queue))
            self.assertFalse(summary["api_credentials_stored"])
            visual_request = next(row for row in queue if row["request_id"] == visual["clip_id"])
            self.assertIn("open_world_findings", visual_request["assessment_dimensions"])
            self.assertIn("findings", visual_request["required_response"])
            self.assertIn("unusable_visual_quality", visual_request["candidate_tasks"])
            self.assertEqual(
                visual_request["trigger_tasks"],
                ["mano_overlay_drift", "semantic_camera_shake"],
            )
            self.assertEqual(visual_request["baseline_qc"]["metrics"], {})

    def test_clean_dataset_still_gets_deterministic_unlabeled_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = create_fixture(root / "dataset", frames=300)
            quality = root / "quality"
            quality.mkdir()
            (quality / "episodes.jsonl").write_text(
                json.dumps({"episode_index": 0, "length": 300}) + "\n",
                encoding="utf-8",
            )
            (quality / "bad_frames.jsonl").write_text("", encoding="utf-8")
            task_config = Path(__file__).parents[1] / "config" / "visual_model_tasks.json"

            first = plan_qc_clips(
                dataset,
                quality,
                root / "first",
                task_config,
                minimum_control_clips=2,
                seed=11,
            )
            second = plan_qc_clips(
                dataset,
                quality,
                root / "second",
                task_config,
                minimum_control_clips=2,
                seed=11,
            )
            first_rows = _read_jsonl(root / "first" / "clip-candidates.jsonl")
            second_rows = _read_jsonl(root / "second" / "clip-candidates.jsonl")

            self.assertEqual(first["selection_counts"], {"deterministic_clean_gap_control": 2})
            self.assertEqual(
                [(row["clip_id"], row["clip_start_s"]) for row in first_rows],
                [(row["clip_id"], row["clip_start_s"]) for row in second_rows],
            )

    def test_bounded_selection_keeps_controls_and_source_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = create_fixture(root / "dataset", frames=900, episodes=3)
            quality = root / "quality"
            quality.mkdir()
            (quality / "episodes.jsonl").write_text(
                "".join(
                    json.dumps({"episode_index": index, "length": 900}) + "\n"
                    for index in range(3)
                ),
                encoding="utf-8",
            )
            events = [
                {
                    "episode_index": episode,
                    "frame_index": frame,
                    "code": "temporal_spike",
                    "severity": "error",
                }
                for episode in range(3)
                for frame in (150, 450, 750)
            ]
            (quality / "bad_frames.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in events),
                encoding="utf-8",
            )
            task_config = Path(__file__).parents[1] / "config" / "visual_model_tasks.json"

            summary = plan_qc_clips(
                dataset,
                quality,
                root / "out",
                task_config,
                maximum_clips=5,
                minimum_control_clips=1,
                control_ratio=0.25,
                source_class="supplier_dataset",
                source_dataset="supplier-batch-a",
                supplier_id="vendor-a",
            )
            clips = _read_jsonl(root / "out" / "clip-candidates.jsonl")
            queue = _read_jsonl(root / "out" / "teacher-api-queue.jsonl")

            self.assertEqual(summary["clips"], 5)
            self.assertTrue(summary["bounded_streaming_selection"])
            self.assertGreaterEqual(summary["produced_random_controls"], 1)
            self.assertEqual({row["source_class"] for row in clips}, {"supplier_dataset"})
            self.assertEqual({row["source_dataset"] for row in queue}, {"supplier-batch-a"})
            self.assertEqual({row["supplier_id"] for row in queue}, {"vendor-a"})
            self.assertTrue(all(row["split_group_source"] == "raw_source_uri" for row in queue))
            self.assertEqual(len({row["request_id"] for row in queue}), len(queue))
            controls = [
                row for row in queue if row["selection_source"].endswith("_control")
            ]
            self.assertTrue(controls)
            self.assertTrue(all(not row["trigger_tasks"] for row in controls))
            self.assertTrue(all("mano_overlay_drift" in row["candidate_tasks"] for row in controls))


if __name__ == "__main__":
    unittest.main()
