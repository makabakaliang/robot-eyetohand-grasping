"""Tests for hand-eye calibration math helpers."""
import math
import numpy as np
import pytest
from hand_eye_calibration import euler_to_rotation_matrix, rotation_matrix_to_euler


def test_rotation_matrix_is_orthonormal():
    R = euler_to_rotation_matrix(10, 20, 30)
    # R^T R = I
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-9)
    # det(R) = 1
    assert math.isclose(np.linalg.det(R), 1.0, abs_tol=1e-9)


def test_euler_roundtrip():
    rx, ry, rz = -15.0, 35.0, 80.0
    R = euler_to_rotation_matrix(rx, ry, rz)
    rx2, ry2, rz2 = rotation_matrix_to_euler(R)
    np.testing.assert_allclose([rx, ry, rz], [rx2, ry2, rz2], atol=1e-6)


def test_zero_euler_returns_identity():
    R = euler_to_rotation_matrix(0, 0, 0)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-9)


def test_single_axis_rotation():
    R = euler_to_rotation_matrix(90, 0, 0)
    # rotating +Y by 90 deg around X should give +Z
    v = np.array([0.0, 1.0, 0.0])
    v2 = R @ v
    np.testing.assert_allclose(v2, [0.0, 0.0, 1.0], atol=1e-9)
