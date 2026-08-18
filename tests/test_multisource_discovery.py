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


if __name__ == "__main__":
    unittest.main()
