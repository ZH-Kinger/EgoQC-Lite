import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from create_fixture import create_fixture
from egoqc.completion import build_completion_overlay, plan_public_completion
from egoqc.provenance import code_version


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(
            (Path(__file__).parents[1] / "config" / "default.json").read_text()
        )

    def test_safe_missing_fields_are_written_to_overlay_without_touching_raw(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = create_fixture(root / "public-dataset")
            source = dataset / "data" / "chunk-000" / "file-000.parquet"
            table = pq.read_table(source)
            removed = {
                "index", "timestamp", "main_type", "state_mask", "intrinsics",
                "left_seg_start", "left_seg_end", "right_seg_start", "right_seg_end",
            }
            pq.write_table(
                table.select([name for name in table.column_names if name not in removed]),
                source,
            )
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            plan_path = root / "completion-plan.json"
            plan = plan_public_completion(dataset, self.config, plan_path)
            file_plan = plan["files"][0]
            statuses = {
                action["field"]: action["status"]
                for action in file_plan["completions"]
            }
            self.assertEqual(statuses["timestamp"], "derived_nominal")
            self.assertEqual(statuses["index"], "derived_exact")
            self.assertEqual(statuses["state_mask"], "derived_exact")
            self.assertEqual(statuses["intrinsics"], "derived_exact")
            self.assertEqual(statuses["main_type"], "defaulted")
            self.assertIn("hand_pose_training", plan["allowed_uses"])
            self.assertFalse(plan["policy"]["nominal_timestamp_is_independent_clock"])

            output = root / "overlay"
            result = build_completion_overlay(dataset, plan_path, output)
            self.assertFalse(result["raw_modified"])
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)
            overlay = pq.read_table(output / result["files"][0]["overlay"])
            self.assertEqual(len(overlay), len(table))
            self.assertEqual(overlay["index"].to_pylist(), list(range(len(table))))
            self.assertEqual(overlay["main_type"].to_pylist(), [-1] * len(table))
            self.assertTrue(np.allclose(overlay["timestamp"].to_numpy(), np.arange(len(table)) / 30.0))
            self.assertEqual(overlay["state_mask"].to_pylist(), [[True, True]] * len(table))
            intrinsics = np.asarray(overlay["intrinsics"].to_pylist())
            self.assertEqual(intrinsics.shape, (len(table), 9))
            self.assertTrue(np.all(intrinsics[:, 0] > 0))
            self.assertTrue(np.all(intrinsics[:, 4] > 0))

    def test_code_version_contains_source_fingerprint(self):
        identity = code_version()
        self.assertIn("+src.", identity)
        self.assertGreaterEqual(len(identity.rsplit("+src.", 1)[1]), 16)

    def test_overlay_refuses_source_changed_after_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = create_fixture(root / "public-dataset")
            plan_path = root / "completion-plan.json"
            plan_public_completion(dataset, self.config, plan_path)
            source = dataset / "data" / "chunk-000" / "file-000.parquet"
            stat = source.stat()
            os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
            with self.assertRaisesRegex(ValueError, "源数据在 plan 后发生变化"):
                build_completion_overlay(dataset, plan_path, root / "overlay")


if __name__ == "__main__":
    unittest.main()
