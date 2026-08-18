import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from egoqc.adapter_clip_selection import plan_adapter_clips


class AdapterClipSelectionTests(unittest.TestCase):
    def test_builds_bounded_teacher_queue_from_canonical_episode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "tasks.json"
            config.write_text(json.dumps({
                "model_tasks": {"hand_absent": {}, "camera_shake": {}},
                "assessment_dimensions": {"visual_quality": "quality"},
            }), encoding="utf-8")
            report = {
                "canonical": {
                    "episode_id": "part1/task/0",
                    "source_format": "egodex_hdf5",
                    "duration_s": 9.6,
                    "video": {"path": "/readonly/0.mp4"},
                    "labels": {"task": "Task"},
                    "capabilities": {"video": True},
                }
            }
            with patch("egoqc.adapter_clip_selection.inspect_adapter", return_value=report):
                summary = plan_adapter_clips(
                    root,
                    "part1/task/0",
                    root / "out",
                    config,
                    window_s=6.0,
                    maximum_clips=3,
                )
            self.assertEqual(summary["teacher_api_requests"], 2)
            rows = [
                json.loads(line)
                for line in (root / "out" / "teacher-api-queue.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(rows), 2)
            self.assertAlmostEqual(rows[0]["clip_start_s"], 0.0)
            self.assertAlmostEqual(rows[0]["clip_end_s"], 6.0)
            self.assertAlmostEqual(rows[1]["clip_start_s"], 3.6)
            self.assertAlmostEqual(rows[1]["clip_end_s"], 9.6)
            self.assertEqual(set(rows[0]["candidate_tasks"]), {"hand_absent", "camera_shake"})
            self.assertTrue(rows[0]["output_path"].endswith("teacher-label.json"))


if __name__ == "__main__":
    unittest.main()
