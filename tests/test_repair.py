import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from create_fixture import create_fixture
from egoqc.math3d import euler_xyz_to_matrix, so3_errors
from egoqc.repair import one_euro_rotations, repair_episode_records, write_repair_preview


class RepairTests(unittest.TestCase):
    def test_rotation_filter_stays_on_so3(self):
        angles = np.zeros((12, 3))
        angles[:, 2] = np.linspace(0.0, 0.5, 12)
        angles[6, 2] += 0.4
        repaired = one_euro_rotations(
            euler_xyz_to_matrix(angles), 30.0, 2.0, 0.7, 1.0
        )
        orthogonality, determinant = so3_errors(repaired)
        self.assertLess(float(orthogonality.max()), 1e-10)
        self.assertLess(float(determinant.max()), 1e-10)

    def test_invalid_gap_is_not_changed_or_bridged(self):
        records = []
        eye = np.eye(3).reshape(-1).tolist()
        pose = np.broadcast_to(np.eye(3), (15, 3, 3)).reshape(-1).tolist()
        for frame in range(7):
            state = np.zeros(122)
            state[51:61] = 3.0
            state[112:122] = 4.0
            records.append(
                {
                    "frame_index": frame,
                    "state_mask": [frame != 3, False],
                    "extrinsics_w2c": np.eye(4).reshape(-1).tolist(),
                    "observation.state": state.tolist(),
                    "left_transl_world": [float(frame if frame < 3 else frame + 100), 0.0, 1.0],
                    "left_orient_world": eye,
                    "left_hand_pose": pose,
                    "right_transl_world": [0.0, 0.0, 1.0],
                    "right_orient_world": eye,
                    "right_hand_pose": pose,
                }
            )
        repaired = repair_episode_records(records, 30.0, {})
        self.assertEqual(repaired[3]["left_transl_world"], records[3]["left_transl_world"])
        self.assertEqual(repaired[3]["observation.state"], records[3]["observation.state"])
        self.assertEqual(repaired[0]["observation.state"][51:61], [3.0] * 10)

    def test_preview_writes_derived_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            dataset = create_fixture(base / "dataset", frames=8)
            output = base / "preview"
            config = json.loads(
                (Path(__file__).parents[1] / "config" / "default.json").read_text()
            )
            summary = write_repair_preview(dataset, 0, output, config)
            self.assertTrue(summary["source_unchanged"])
            self.assertFalse(summary["source_motion_acceptance"]["repaired_preview_can_change_acceptance"])
            self.assertEqual(pq.read_table(output / "repair-preview.parquet").num_rows, 8)
            self.assertTrue((output / "repair-deltas.parquet").exists())
            self.assertTrue((output / "repair-metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
