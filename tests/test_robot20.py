from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from egoqc.robot20 import inspect_urdf


MINIMAL = """<?xml version="1.0"?>
<robot name="test">
  <link name="palm"/><link name="finger"/>
  <joint name="finger_joint" type="revolute">
    <parent link="palm"/><child link="finger"/><axis xyz="0 1 0"/>
    <limit lower="-1" upper="1" effort="1" velocity="2"/>
  </joint>
</robot>
"""


class Robot20Tests(unittest.TestCase):
    def test_minimal_urdf(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "hand.urdf"
            path.write_text(MINIMAL, encoding="utf-8")
            report = inspect_urdf(path, expected_dof=1)
            self.assertTrue(report["ok"])
            self.assertEqual(report["joint_order"], ["finger_joint"])

    def test_invalid_limit_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "hand.urdf"
            path.write_text(MINIMAL.replace('lower="-1" upper="1"', 'lower="2" upper="1"'), encoding="utf-8")
            report = inspect_urdf(path, expected_dof=1)
            self.assertFalse(report["ok"])
            self.assertTrue(any("lower >= upper" in value for value in report["errors"]))

if __name__ == "__main__":
    unittest.main()
