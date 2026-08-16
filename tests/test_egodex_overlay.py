import unittest

import numpy as np

from egoqc.egodex_overlay import _project


class EgoDexOverlayTests(unittest.TestCase):
    def test_official_pinhole_projection(self):
        intrinsic = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
        self.assertEqual(_project(np.array([1.0, 2.0, 10.0]), intrinsic), (60.0, 60.0))

    def test_point_behind_camera_is_not_drawn(self):
        self.assertIsNone(_project(np.array([0.0, 0.0, -1.0]), np.eye(3)))


if __name__ == "__main__":
    unittest.main()
