import json
import tempfile
import unittest
from pathlib import Path

from egoqc.live_server import live_payload


class LiveServerTests(unittest.TestCase):
    def test_reads_atomic_live_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / "live/runs/run-a"
            (run_root / "shards").mkdir(parents=True)
            (root / "live/current.json").write_text(
                json.dumps({"run_id": "run-a", "status": "running", "fraction": 0.5})
            )
            (run_root / "shards/a.jsonl").write_text(
                json.dumps({"episode_index": 3, "tier": "bronze", "provisional": True}) + "\n"
            )
            payload = live_payload(root)
            self.assertEqual(payload["episode_count"], 1)
            self.assertEqual(payload["episodes"][0]["episode_index"], 3)
            self.assertEqual(payload["status"]["status"], "running")


if __name__ == "__main__":
    unittest.main()
