"""Pose file I/O and small coordinate helpers."""
import numpy as np
from pathlib import Path


def load_pose_list(path):
    """Load a pose file with one (x,y,z,rx,ry,rz) line per row.

    Empty lines and lines starting with '#' are ignored.
    Returns an (N, 6) ndarray.
    """
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append([float(x) for x in line.split()])
    return np.asarray(rows, dtype=float)


def pose_to_matrix(xyz, rpy_deg):
    """Compose a 4x4 homogeneous matrix from translation and euler (deg)."""
    from hand_eye_calibration import euler_to_rotation_matrix
    T = np.eye(4)
    T[:3, :3] = euler_to_rotation_matrix(*rpy_deg)
    T[:3, 3] = xyz
    return T


def transform_points(T, points):
    """Apply 4x4 transform to (N, 3) points."""
    points = np.asarray(points, dtype=float)
    homog = np.hstack([points, np.ones((len(points), 1))])
    return (T @ homog.T).T[:, :3]
