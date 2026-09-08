"""Quick 3D scatter of captured calibration poses for sanity checking."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pose_utils import load_pose_list


def plot(cam_file, robot_file, out="poses_preview.png"):
    cam = load_pose_list(cam_file)
    robot = load_pose_list(robot_file)
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(cam[:, 0], cam[:, 1], cam[:, 2], c="tab:blue")
    ax1.set_title("AprilTag poses (camera frame)")
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(robot[:, 0], robot[:, 1], robot[:, 2], c="tab:red")
    ax2.set_title("Robot base poses")
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    print(f"saved {out}")


if __name__ == "__main__":
    plot("apriltag_poses.txt", "robot_base_poses.txt")
