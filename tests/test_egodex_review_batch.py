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


if __name__ == "__main__":
    unittest.main()
