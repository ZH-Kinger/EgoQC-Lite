import unittest

import numpy as np

from egoqc.math3d import (
    axis_angle_to_matrix,
    euler_xyz_to_matrix,
    geodesic_degrees,
    longest_false_run,
    matrix_to_euler_xyz,
    so3_errors,
    transform_points,
)


class Math3DTests(unittest.TestCase):
    def test_identity_rotation(self):
        rotation = euler_xyz_to_matrix(np.zeros((2, 3)))
        np.testing.assert_allclose(rotation, np.broadcast_to(np.eye(3), (2, 3, 3)))
        orth, det = so3_errors(rotation)
        np.testing.assert_allclose(orth, 0.0)
        np.testing.assert_allclose(det, 0.0)

    def test_axis_angle(self):
        rotation = axis_angle_to_matrix(np.array([[0.0, 0.0, np.pi / 2]]))
        expected = euler_xyz_to_matrix(np.array([[0.0, 0.0, np.pi / 2]]))
        np.testing.assert_allclose(rotation, expected, atol=1e-10)

    def test_geodesic(self):
        identity = np.eye(3)[None]
        quarter_turn = euler_xyz_to_matrix(np.array([[0.0, 0.0, np.pi / 2]]))
        np.testing.assert_allclose(geodesic_degrees(identity, quarter_turn), 90.0)

    def test_euler_matrix_roundtrip(self):
        values = np.array([[0.2, -0.4, 0.6], [-1.0, 0.3, 2.0]])
        rotations = euler_xyz_to_matrix(values)
        reconstructed = euler_xyz_to_matrix(matrix_to_euler_xyz(rotations))
        np.testing.assert_allclose(reconstructed, rotations, atol=1e-10)

    def test_transform(self):
        extrinsic = np.eye(4)[None]
        extrinsic[0, :3, 3] = [1, 2, 3]
        point = np.array([[2, 3, 4]])
        np.testing.assert_allclose(transform_points(extrinsic, point), [[3, 5, 7]])

    def test_longest_gap(self):
        self.assertEqual(longest_false_run(np.array([True, False, False, True, False])), 2)


if __name__ == "__main__":
    unittest.main()
