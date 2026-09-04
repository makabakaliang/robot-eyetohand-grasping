"""Evaluate hand-eye calibration quality by reprojecting paired poses."""
import numpy as np
from pose_utils import load_pose_list, pose_to_matrix


def evaluate(T_base_cam, cam_poses_file, robot_poses_file):
    """Compute RMS residual between observed and predicted camera poses.

    Returns (rms_translation_mm, max_translation_mm).
    """
    cam = load_pose_list(cam_poses_file)
    robot = load_pose_list(robot_poses_file)
    n = min(len(cam), len(robot))
    errors = []
    for i in range(n):
        T_robot = pose_to_matrix(robot[i, :3], robot[i, 3:6])
        T_cam = pose_to_matrix(cam[i, :3], cam[i, 3:6])
        # Predicted camera frame in robot base: T_robot * T_base_cam
        T_pred = T_robot @ T_base_cam
        err = T_pred[:3, 3] - T_cam[:3, 3]
        errors.append(np.linalg.norm(err))
    errors = np.asarray(errors)
    return float(np.sqrt((errors ** 2).mean())), float(errors.max())


def main():
    T = np.load("T_base_cam.npy") if __import__("pathlib").Path("T_base_cam.npy").exists() else np.eye(4)
    rms, mx = evaluate(T, "apriltag_poses.txt", "robot_base_poses.txt")
    print(f"calibration RMS = {rms:.2f} mm, max = {mx:.2f} mm")


if __name__ == "__main__":
    main()
