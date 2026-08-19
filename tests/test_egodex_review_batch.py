import json
import tempfile
import unittest
from pathlib import Path

from egoqc.egodex_review_batch import build_egodex_review_batch


class EgoDexReviewBatchTests(unittest.TestCase):
    def test_builds_balanced_raw_review_queue_without_overlay_task(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            profiles = root / "profiles.jsonl"
            output = root / "derived"
            config = root / "tasks.json"
            rows = []
            for index, tier in enumerate(("candidate-clean", "hard-negative", "review", "review")):
                rows.append({
                    "episode_id": f"part1/task-{index % 2}/{index}",
                    "partition": "part1",
                    "task": f"task-{index % 2}",
                    "hdf5_path": str(dataset / "part1" / f"task-{index % 2}" / f"{index}.hdf5"),
                    "video_path": str(dataset / "part1" / f"task-{index % 2}" / f"{index}.mp4"),
                    "duration_s": 10.0,
                    "candidate_tier": tier,
                    "annotation_score": 0.5,
                    "hard_gates": {},
                    "hand_metrics": {},
                    "labels": {"task": "test"},
                    "capabilities": {"video": True},
                })
            profiles.write_text("".join(json.dumps(row) + "\n" for row in rows))
            config.write_text(json.dumps({
                "assessment_dimensions": {"visual_quality": "quality"},
                "model_tasks": {
                    "hand_absent": {},
                    "mano_overlay_drift": {},
                },
            }))
            summary = build_egodex_review_batch(
                profiles, config, output, dataset=dataset, maximum_review=1
            )
            self.assertEqual(summary["items"], 3)
            self.assertEqual(summary["tier_counts"]["review"], 1)
            teacher = [
                json.loads(line)
                for line in (output / "teacher-api-queue.jsonl").read_text().splitlines()
            ]
            self.assertEqual(teacher[0]["candidate_tasks"], ["hand_absent"])
            self.assertTrue(all(row["source_readonly"] for row in teacher))
            self.assertFalse(summary["teacher_api_called"])

    def test_targets_longest_hand_absence_window(self):
        import h5py
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            hdf5_path = dataset / "part1/task/1.hdf5"
            hdf5_path.parent.mkdir(parents=True)
            with h5py.File(hdf5_path, "w") as handle:
                confidences = handle.create_group("confidences")
                values = np.ones(300, dtype=np.float32)
                values[120:210] = 0.0
                confidences.create_dataset("leftHand", data=values)
                confidences.create_dataset("rightHand", data=values)
            row = {
                "episode_id": "part1/task/1",
                "partition": "part1",
                "task": "task",
                "hdf5_path": str(hdf5_path),
                "video_path": str(dataset / "part1/task/1.mp4"),
                "duration_s": 10.0,
                "video": {"fps": 30.0},
                "candidate_tier": "programmatic-reject",
                "annotation_score": 0.2,
                "hard_gates": {
                    "duration_at_least_minimum": True,
                    "fps_at_least_minimum": True,
                    "resolution_at_least_720p": True,
                    "continuous_hand_visibility_at_least_minimum": True,
                    "hand_absence_not_over_limit": False,
                    "no_audio": True,
                },
                "hand_metrics": {"longest_absence_s": 3.0},
                "labels": {},
                "capabilities": {"video": True},
            }
            profiles = root / "profiles.jsonl"
            profiles.write_text(json.dumps(row) + "\n")
            config = root / "tasks.json"
            config.write_text(json.dumps({
                "assessment_dimensions": {},
                "model_tasks": {"hand_absent": {}},
            }))

            summary = build_egodex_review_batch(
                profiles,
                config,
                root / "derived",
                dataset=dataset,
                maximum_clean=0,
                maximum_hard_negative=0,
                maximum_review=0,
                maximum_hand_absence=1,
                window_s=6.0,
            )
            teacher = json.loads((root / "derived/teacher-api-queue.jsonl").read_text())

            self.assertEqual(summary["tier_counts"]["targeted-hand-absence"], 1)
            self.assertEqual(teacher["trigger_tasks"], ["hand_absent"])
            self.assertLess(teacher["clip_start_s"], 5.5)
            self.assertGreater(teacher["clip_end_s"], 4.0)


if __name__ == "__main__":
    unittest.main()
