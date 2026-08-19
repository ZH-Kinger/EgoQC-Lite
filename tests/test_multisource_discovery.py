import json
import tempfile
import unittest
from pathlib import Path

from egoqc.multisource_discovery import discover_lerobot_roots


class MultisourceDiscoveryTests(unittest.TestCase):
    def test_discovers_task_balanced_lerobot_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "source"
            output = base / "output"
            for task in ("pick", "place"):
                for index in range(3):
                    meta = source / task / str(index) / "meta"
                    meta.mkdir(parents=True)
                    (meta / "info.json").write_text(json.dumps({
                        "codebase_version": "v3.0",
                        "robot_type": "ego_hand",
                        "total_episodes": 1,
                        "total_frames": 300,
                        "fps": 30.0,
                        "features": {
                            "observation.images.ego": {},
                            "observation.state": {},
                            "state_mask": {},
                            "intrinsics": {},
                            "extrinsics_w2c": {},
                        },
                    }))
            summary = discover_lerobot_roots(
                source, output, maximum_per_task=2, workers=4
            )
            self.assertEqual(summary["dataset_roots"], 4)
            self.assertEqual(summary["task_groups"], 2)
            self.assertEqual(summary["episode_hints"], 4)
            self.assertFalse(summary["capability_counts"]["raw_world_hands"])
            self.assertEqual(
                len((output / "dataset-list.txt").read_text().splitlines()), 4
            )

    def test_ignores_non_directory_entries_in_task_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "source"
            output = base / "output"
            task = source / "pick"
            task.mkdir(parents=True)
            (task / "README.txt").write_text("not a dataset")
            meta = task / "dataset" / "meta"
            meta.mkdir(parents=True)
            (meta / "info.json").write_text(json.dumps({
                "total_episodes": 1,
                "total_frames": 30,
                "fps": 30.0,
                "features": {"observation.images.ego": {}},
            }))

            summary = discover_lerobot_roots(
                source, output, maximum_per_task=2, workers=2
            )

            self.assertEqual(summary["dataset_roots"], 1)
            self.assertEqual(summary["errors"], 0)

    def test_can_bound_number_of_task_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "source"
            output = base / "output"
            for task in ("a", "b", "c", "d"):
                meta = source / task / "dataset" / "meta"
                meta.mkdir(parents=True)
                (meta / "info.json").write_text(json.dumps({
                    "total_episodes": 1,
                    "total_frames": 30,
                    "fps": 30.0,
                    "features": {"observation.images.ego": {}},
                }))

            summary = discover_lerobot_roots(
                source,
                output,
                maximum_per_task=1,
                maximum_tasks=2,
                workers=2,
                seed=99,
            )

            self.assertEqual(summary["available_task_groups"], 4)
            self.assertEqual(summary["task_groups"], 2)
            self.assertEqual(summary["dataset_roots"], 2)


if __name__ == "__main__":
    unittest.main()
