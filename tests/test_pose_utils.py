"""Tests for pose_utils helpers."""
import numpy as np
from pose_utils import pose_to_matrix, transform_points


def test_pose_to_matrix_shape():
    T = pose_to_matrix([10, 20, 30], [0, 0, 0])
    assert T.shape == (4, 4)
    np.testing.assert_allclose(T[:3, 3], [10, 20, 30])
    np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-9)


def test_transform_points_translation():
    T = np.eye(4)
    T[:3, 3] = [100, 0, 0]
    out = transform_points(T, [[0, 0, 0], [10, 20, 30]])
    np.testing.assert_allclose(out, [[100, 0, 0], [110, 20, 30]])
