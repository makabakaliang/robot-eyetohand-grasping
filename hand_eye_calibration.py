#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手眼标定算法
计算相机到机械臂基座的变换矩阵
手眼标定算法，计算T_base_cam(机械臂基座到相机的变换矩阵，保存为.npy和txt格式)
"""

import numpy as np
import math
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
FILE_CAM = BASE_DIR / "apriltag_poses.txt"
FILE_ROBOT = BASE_DIR / "robot_base_poses.txt"
OUT_NPY = BASE_DIR / "T_base_cam.npy"
OUT_TXT = BASE_DIR / "T_base_cam.txt"


def euler_to_rotation_matrix(rx, ry, rz):
    """
    将欧拉角（度）转换为旋转矩阵
    使用 Z-Y-X 顺序 (Tait-Bryan angles)

    参数:
        rx, ry, rz: 绕X、Y、Z轴的旋转角（度）

    返回:
        3x3 旋转矩阵
    """
    rx_rad = math.radians(rx)
    ry_rad = math.radians(ry)
    rz_rad = math.radians(rz)

    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(rx_rad), -math.sin(rx_rad)],
        [0, math.sin(rx_rad), math.cos(rx_rad)]
    ])

    Ry = np.array([
        [math.cos(ry_rad), 0, math.sin(ry_rad)],
        [0, 1, 0],
        [-math.sin(ry_rad), 0, math.cos(ry_rad)]
    ])

    Rz = np.array([
        [math.cos(rz_rad), -math.sin(rz_rad), 0],
        [math.sin(rz_rad), math.cos(rz_rad), 0],
        [0, 0, 1]
    ])

    R = Rz @ Ry @ Rx
    return R


def rotation_matrix_to_euler(R):
    """
    将旋转矩阵转换为欧拉角（度）
    使用 Z-Y-X 顺序

    参数:
        R: 3x3 旋转矩阵

    返回:
        rx, ry, rz: 欧拉角（度）
    """
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        rx = math.atan2(R[2, 1], R[2, 2])
        ry = math.atan2(-R[2, 0], sy)
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        rx = math.atan2(-R[1, 2], R[1, 1])
        ry = math.atan2(-R[2, 0], sy)
        rz = 0

    return math.degrees(rx), math.degrees(ry), math.degrees(rz)


def pose_to_transform_matrix(pose):
    """
    将位姿转换为4x4齐次变换矩阵

    参数:
        pose: (x, y, z, rx, ry, rz)

    返回:
        4x4 齐次变换矩阵
    """
    x, y, z, rx, ry, rz = pose
    R = euler_to_rotation_matrix(rx, ry, rz)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def transform_matrix_to_pose(T):
    """
    将4x4齐次变换矩阵转换为位姿

    参数:
        T: 4x4 齐次变换矩阵

    返回:
        (x, y, z, rx, ry, rz)
    """
    x, y, z = T[:3, 3]
    R = T[:3, :3]
    rx, ry, rz = rotation_matrix_to_euler(R)
    return (x, y, z, rx, ry, rz)


def parse_pose_line(line, file_name, line_no):
    """
    解析一行位姿数据。
    新格式：X,Y,Z,Rx,Ry,Rz（6列）
    旧格式兼容：序号,X,Y,Z,Rx,Ry,Rz（7列，会自动忽略第1列序号）
    """
    s = line.strip()
    if not s or s.startswith(('#', '//')):
        return None

    parts = [p.strip() for p in s.split(',')]
    if len(parts) == 7:
        # 兼容旧版GUI写出的“序号 + 6个坐标”格式
        parts = parts[1:]
    elif len(parts) != 6:
        raise ValueError(
            f"{file_name} 第 {line_no} 行格式错误：需要6个数 "
            f"X,Y,Z,Rx,Ry,Rz；当前为 {len(parts)} 列：{s}"
        )

    try:
        return tuple(float(x) for x in parts)
    except ValueError as exc:
        raise ValueError(f"{file_name} 第 {line_no} 行存在非数字内容：{s}") from exc


def read_pose_file(path, label):
    poses = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, start=1):
                pose = parse_pose_line(line, Path(path).name, line_no)
                if pose is not None:
                    poses.append(pose)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"未找到{label}文件：{path}") from exc

    print(f"✓ {label}: {len(poses)} 个 ({path})")
    return poses


def rigid_transform_3d_svd(points_a, points_b):
    """
    使用SVD方法计算刚体变换（Kabsch算法）
    求解: points_b = R @ points_a + t

    参数:
        points_a: Nx3 点集A（源点集）
        points_b: Nx3 点集B（目标点集）

    返回:
        R: 3x3 旋转矩阵
        t: 3x1 平移向量
    """
    assert points_a.shape == points_b.shape, "点集维度必须相同"

    # 计算质心
    centroid_a = np.mean(points_a, axis=0)
    centroid_b = np.mean(points_b, axis=0)

    # 去中心化
    aa = points_a - centroid_a
    bb = points_b - centroid_b

    # 计算协方差矩阵 H = A^T @ B
    H = aa.T @ bb

    # SVD分解
    U, S, Vt = np.linalg.svd(H)

    # 计算旋转矩阵
    R = Vt.T @ U.T

    # 处理反射情况（确保行列式为1）
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 计算平移向量
    t = centroid_b - R @ centroid_a

    return R, t


def hand_eye_calibration_svd(poses_cam, poses_base):
    """
    使用SVD方法进行手眼标定
    求解相机到基座的变换矩阵 T_base_cam
    使得: P_base = T_base_cam @ P_cam

    参数:
        poses_cam: 相机坐标系下的位姿列表
        poses_base: 基座坐标系下的位姿列表

    返回:
        T_base_cam: 4x4 变换矩阵
    """
    n = len(poses_cam)
    assert n == len(poses_base), "位姿数量必须相同"
    assert n >= 3, "至少需要3个位姿点"

    # 提取位置点
    points_cam = np.array([pose[:3] for pose in poses_cam])
    points_base = np.array([pose[:3] for pose in poses_base])

    # 使用SVD求解刚体变换
    R, t = rigid_transform_3d_svd(points_cam, points_base)

    # 构建齐次变换矩阵
    T_base_cam = np.eye(4)
    T_base_cam[:3, :3] = R
    T_base_cam[:3, 3] = t

    return T_base_cam


def rotation_error_deg(R_gt, R_pred):
    """Compute rotation error between two rotation matrices in degrees (0~180)."""
    R_err = R_gt.T @ R_pred
    c = (np.trace(R_err) - 1.0) / 2.0
    c = float(np.clip(c, -1.0, 1.0))
    return math.degrees(math.acos(c))


def evaluate_calibration(T_base_cam, poses_cam, poses_base):
    """
    评估标定结果

    参数:
        T_base_cam: 标定得到的变换矩阵
        poses_cam: 相机坐标系下的位姿
        poses_base: 基座坐标系下的位姿（真值）

    返回:
        平均位置误差、平均旋转误差
    """
    position_errors = []
    rotation_errors = []

    for pose_cam, pose_base_gt in zip(poses_cam, poses_base):
        # 将相机坐标系位姿转换到基座坐标系
        T_cam = pose_to_transform_matrix(pose_cam)
        T_base_pred = T_base_cam @ T_cam
        pose_base_pred = transform_matrix_to_pose(T_base_pred)

        # 计算位置误差
        pos_error = np.linalg.norm(
            np.array(pose_base_pred[:3]) - np.array(pose_base_gt[:3])
        )
        position_errors.append(pos_error)

        # 计算旋转误差（推荐：用旋转矩阵夹角，避免欧拉角±180°跳变导致的“假大误差”）
        R_pred = T_base_pred[:3, :3]
        R_gt = pose_to_transform_matrix(pose_base_gt)[:3, :3]
        rot_error = rotation_error_deg(R_gt, R_pred)
        rotation_errors.append(rot_error)

    avg_pos_error = np.mean(position_errors)
    avg_rot_error = np.mean(rotation_errors)
    max_pos_error = np.max(position_errors)
    max_rot_error = np.max(rotation_errors)

    return avg_pos_error, avg_rot_error, max_pos_error, max_rot_error


def main():
    """主函数"""
    print("=" * 80)
    print("手眼标定算法 - 求解相机到机械臂基座的变换矩阵")
    print("=" * 80)

    # 读取数据
    print("\n【步骤1】读取位姿数据")
    print("-" * 80)

    poses_cam = []
    poses_base = []

    # 读取相机坐标系位姿
    poses_cam = read_pose_file(FILE_CAM, "相机坐标系位姿")

    # 读取基座坐标系位姿
    poses_base = read_pose_file(FILE_ROBOT, "基座坐标系位姿")

    if len(poses_cam) != len(poses_base):
        raise ValueError(f"位姿数量不匹配：视觉点 {len(poses_cam)} 个，机械臂点 {len(poses_base)} 个。请保证两份文件按行一一对应。")

    if len(poses_cam) < 3:
        raise ValueError(f"至少需要3个位姿点进行标定，当前只有 {len(poses_cam)} 个。")

    # 使用SVD方法进行手眼标定
    print("\n【步骤2】使用SVD方法进行手眼标定")
    print("-" * 80)

    T_base_cam = hand_eye_calibration_svd(poses_cam, poses_base)
    pose_result = transform_matrix_to_pose(T_base_cam)

    print("SVD方法标定结果:")
    print(f"  平移 (mm): Tx={pose_result[0]:.3f}, Ty={pose_result[1]:.3f}, Tz={pose_result[2]:.3f}")
    print(f"  旋转 (度): Rx={pose_result[3]:.3f}, Ry={pose_result[4]:.3f}, Rz={pose_result[5]:.3f}")
    print("\n变换矩阵 T_base_cam:")
    print(T_base_cam)

    # 评估标定精度
    print("\n【步骤3】评估标定精度")
    print("-" * 80)

    avg_pos_error, avg_rot_error, max_pos_error, max_rot_error = evaluate_calibration(
        T_base_cam, poses_cam, poses_base
    )

    print("标定精度评估:")
    print(f"  平均位置误差: {avg_pos_error:.6f} mm")
    print(f"  最大位置误差: {max_pos_error:.6f} mm")
    print(f"  平均旋转误差: {avg_rot_error:.6f}°")
    print(f"  最大旋转误差: {max_rot_error:.6f}°")

    # 保存结果
    print("\n【步骤4】保存标定结果")
    print("-" * 80)

    # 保存为numpy格式
    np.save(OUT_NPY, T_base_cam)
    print(f"✓ 变换矩阵已保存至: {OUT_NPY}")

    # 保存为文本格式
    with open(OUT_TXT, 'w', encoding='utf-8') as f:
        f.write(f"# 相机到机械臂基座的变换矩阵 (SVD方法)\n")
        f.write(f"# 平移 (mm): Tx, Ty, Tz\n")
        f.write(f"# 旋转 (度): Rx, Ry, Rz\n")
        f.write(f"{pose_result[0]:.6f},{pose_result[1]:.6f},{pose_result[2]:.6f},")
        f.write(f"{pose_result[3]:.6f},{pose_result[4]:.6f},{pose_result[5]:.6f}\n")
        f.write(f"\n# 齐次变换矩阵 (4x4):\n")
        for row in T_base_cam:
            f.write(f"# {row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}, {row[3]:.6f}\n")

        # ----------------- 新增部分：将精度评估结果写入 txt -----------------
        f.write(f"\n# ----------------------------------------\n")
        f.write(f"# 标定精度评估:\n")
        f.write(f"# 平均位置误差: {avg_pos_error:.6f} mm\n")
        f.write(f"# 最大位置误差: {max_pos_error:.6f} mm\n")
        f.write(f"# 平均旋转误差: {avg_rot_error:.6f}°\n")
        f.write(f"# 最大旋转误差: {max_rot_error:.6f}°\n")
        f.write(f"# ----------------------------------------\n")
        # --------------------------------------------------------------------

    print(f"✓ 变换参数已保存至: {OUT_TXT}")

    print("\n" + "=" * 80)
    print("手眼标定完成！")
    print("=" * 80)


    # 给GUI明确的成功返回值，避免GUI把默认返回值 None 误判为失败
    return True


if __name__ == "__main__":
    main()