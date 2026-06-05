#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aruco标记检测程序
使用RealSense相机检测aruco标记并测量位姿
手眼标定程序（按s保存数据到txt）
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import math


class ArucoPoseDetector:
    """Aruco标记位姿检测器"""

    def __init__(self, marker_size=0.134, camera_width=1280, camera_height=720, fps=6):
        """
        初始化Aruco检测器

        参数:
            marker_size: aruco标记的实际尺寸（米）
            camera_width: 相机宽度，默认1280
            camera_height: 相机高度，默认720
            fps: 相机帧率，默认6
        """
        self.marker_size = marker_size
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.fps = fps

        # 初始化RealSense相机
        self.pipeline = None
        self.config = None
        self.profile = None

        # 相机内参矩阵和畸变系数
        self.camera_matrix = None
        self.dist_coeffs = None

        # Aruco检测器
        self.aruco_dict = None
        self.parameters = None
        self.detector = None

        # 标记位姿存储
        self.poses = []

    def start_camera(self):
        """启动RealSense相机"""
        try:
            # 创建管道和配置
            self.pipeline = rs.pipeline()
            self.config = rs.config()

            # 配置相机参数
            self.config.enable_stream(
                rs.stream.color,
                self.camera_width,
                self.camera_height,
                rs.format.bgr8,
                self.fps
            )
            self.config.enable_stream(
                rs.stream.depth,
                self.camera_width,
                self.camera_height,
                rs.format.z16,
                self.fps
            )

            # 启动相机
            self.profile = self.pipeline.start(self.config)
            print(f"✓ 相机启动成功: {self.camera_width}x{self.camera_height} @ {self.fps}fps")

            # 丢弃前10帧以稳定曝光
            for _ in range(10):
                self.pipeline.wait_for_frames()

            return True

        except Exception as e:
            print(f"✗ 相机启动失败: {e}")
            return False

    def get_camera_parameters(self):
        """获取相机内外参数"""
        try:
            # 【关键修复】获取彩色流配置（而非深度流）
            # ArUco检测是在彩色图像上进行的，必须使用彩色相机的内参
            color_profile = self.profile.get_stream(rs.stream.color)
            color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

            # 构建相机内参矩阵
            self.camera_matrix = np.array([
                [color_intrinsics.fx, 0, color_intrinsics.ppx],
                [0, color_intrinsics.fy, color_intrinsics.ppy],
                [0, 0, 1]
            ])

            # 畸变系数
            self.dist_coeffs = np.array(color_intrinsics.coeffs)

            print("✓ 相机内参获取成功 (彩色流)")
            print(f"  焦距 (fx, fy): ({color_intrinsics.fx:.2f}, {color_intrinsics.fy:.2f})")
            print(f"  主点 (cx, cy): ({color_intrinsics.ppx:.2f}, {color_intrinsics.ppy:.2f})")
            print(f"  畸变系数: {self.dist_coeffs}")

            return True

        except Exception as e:
            print(f"✗ 获取相机参数失败: {e}")
            return False

    def setup_aruco_detector(self):
        """设置Aruco检测器"""
        try:
            # 使用DICT_APRILTAG_36h11字典 (支持AprilTag 36h11标记)
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
            self.parameters = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)

            print("✓ Aruco检测器设置完成 (AprilTag 36h11)")

            # 设置检测参数以提高稳定性
            self.parameters.cornerRefinementMaxIterations = 30
            self.parameters.cornerRefinementMinAccuracy = 0.1
            self.parameters.minCornerDistanceRate = 0.05
            self.parameters.minMarkerDistanceRate = 0.05
            self.parameters.polygonalApproxAccuracyRate = 0.03

            return True

        except Exception as e:
            print(f"✗ Aruco检测器设置失败: {e}")
            return False

    def detect_markers(self, color_image):
        """
        检测aruco标记并计算位姿

        参数:
            color_image: 输入的彩色图像

        返回:
            corners: 检测到的标记角点
            ids: 标记ID
            poses: 位姿信息列表
        """
        try:
            # 检测标记
            corners, ids, rejected = self.detector.detectMarkers(color_image)

            poses = []

            if ids is not None and len(ids) > 0:
                # 估计每个标记的位姿
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners,
                    self.marker_size,
                    self.camera_matrix,
                    self.dist_coeffs
                )

                # 处理每个标记的位姿
                for i, (corner, marker_id, rvec, tvec) in enumerate(zip(corners, ids, rvecs, tvecs)):
                    # 计算旋转矩阵
                    rmat, _ = cv2.Rodrigues(rvec)

                    # 转换为欧拉角 (弧度)
                    euler_angles = self.rotation_matrix_to_euler_angles(rmat)

                    # 转换为度
                    euler_angles_deg = np.degrees(euler_angles)

                    # 提取位置信息（转换为米）
                    position = tvec[0]

                    pose_info = {
                        'id': int(marker_id),
                        'position': position,  # [x, y, z] 米
                        'rotation_vector': rvec[0],  # 旋转向量
                        'rotation_matrix': rmat,  # 旋转矩阵
                        'euler_angles': euler_angles_deg,  # 欧拉角 [Rx, Ry, Rz] 度
                        'corners': corner[0],  # 四个角点坐标
                    }

                    poses.append(pose_info)

                    # 打印位姿信息
                    print(f"\n标记 ID: {marker_id}")
                    print(
                        f"  位置 (mm): X={position[0] * 1000:.1f}, Y={position[1] * 1000:.1f}, Z={position[2] * 1000:.1f}")
                    print(
                        f"  旋转角 (度): Rx={euler_angles_deg[0]:.1f}, Ry={euler_angles_deg[1]:.1f}, Rz={euler_angles_deg[2]:.1f}")

            return corners, ids, poses

        except Exception as e:
            print(f"✗ 标记检测失败: {e}")
            return None, None, []

    def rotation_matrix_to_euler_angles(self, rmat):
        """
        将旋转矩阵转换为欧拉角（修复版）

        修复说明：
        - ArUco标记的Z轴可能指向远离相机，导致Rx接近±180°
        - 通过规范化Z轴朝向（强制指向相机），确保欧拉角的直观性
        - 参考detect.py中的坐标系规范化逻辑
        """
        # 【关键修复】规范化Z轴朝向
        # ArUco标记坐标系：Z轴垂直于标记表面
        # 约束：强制Z轴指向相机（Z分量为正，因为相机坐标系中深度为正）
        z_axis = rmat[:, 2]  # 第三列为Z轴

        if z_axis[2] < 0:  # 如果Z轴指向远离相机
            # 翻转Z轴：等价于绕X轴旋转180°
            flip_matrix = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ])
            rmat = rmat @ flip_matrix

        # 使用Tait-Bryan角序列 (Z-Y-X) 计算欧拉角
        sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(rmat[2, 1], rmat[2, 2])
            y = math.atan2(-rmat[2, 0], sy)
            z = math.atan2(rmat[1, 0], rmat[0, 0])
        else:
            x = math.atan2(-rmat[1, 2], rmat[1, 1])
            y = math.atan2(-rmat[2, 0], sy)
            z = 0

        return np.array([x, y, z])

    def draw_axis(self, image, corners, ids, poses):
        """
        在图像上绘制坐标轴和标记框

        参数:
            image: 输入图像
            corners: 标记角点
            ids: 标记ID
            poses: 位姿信息
        """
        if ids is not None and len(ids) > 0:
            # 绘制标记边界
            cv2.aruco.drawDetectedMarkers(image, corners, ids)

            # 为每个标记绘制坐标轴
            for pose_info in poses:
                rvec = pose_info['rotation_vector'].reshape(1, 1, 3)
                tvec = pose_info['position'].reshape(1, 1, 3)

                # 绘制坐标轴 (长度：标记尺寸的一半)
                axis_length = self.marker_size * 0.5
                cv2.drawFrameAxes(
                    image,
                    self.camera_matrix,
                    self.dist_coeffs,
                    rvec,
                    tvec,
                    axis_length
                )

                # 在标记中心绘制ID
                center = np.mean(pose_info['corners'], axis=0).astype(int)
                cv2.putText(
                    image,
                    f"ID: {pose_info['id']}",
                    (center[0], center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

    def run_detection(self):
        """运行实时检测循环"""
        print("\n" + "=" * 60)
        print("Aruco标记位姿检测器")
        print("=" * 60)

        # 启动相机
        if not self.start_camera():
            return

        # 获取相机参数
        if not self.get_camera_parameters():
            return

        # 设置aruco检测器
        if not self.setup_aruco_detector():
            return

        print("\n开始实时检测...")
        print("按 'q' 退出，按 's' 保存当前帧")

        try:
            frame_count = 0
            while True:
                # 等待帧
                frames = self.pipeline.wait_for_frames()

                # 获取彩色和深度帧
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                # 转换为numpy数组
                color_image = np.asanyarray(color_frame.get_data())

                # 检测aruco标记
                corners, ids, poses = self.detect_markers(color_image)

                # 绘制结果
                self.draw_axis(color_image, corners, ids, poses)

                # 显示信息
                frame_count += 1
                cv2.putText(
                    color_image,
                    f"Frame: {frame_count} | Markers: {len(ids) if ids is not None else 0}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

                # 显示图像
                cv2.imshow('Aruco Pose Detection', color_image)

                # 按键处理
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n退出检测...")
                    break
                elif key == ord('s'):
                    # 保存位姿数据到txt文件
                    if poses and len(poses) > 0:
                        pose_file = "apriltag_poses.txt"
                        with open(pose_file, 'a', encoding='utf-8') as f:
                            for pose_info in poses:
                                x = pose_info['position'][0] * 1000  # 转换为mm
                                y = pose_info['position'][1] * 1000
                                z = pose_info['position'][2] * 1000
                                rx = pose_info['euler_angles'][0]
                                ry = pose_info['euler_angles'][1]
                                rz = pose_info['euler_angles'][2]
                                f.write(f"{x:.3f},{y:.3f},{z:.3f},{rx:.3f},{ry:.3f},{rz:.3f}\n")
                        print(f"✓ 位姿数据已保存至: {pose_file}")
                    else:
                        print("⚠️ 未检测到标记，无法保存")

        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            self.stop_camera()

    def stop_camera(self):
        """停止相机"""
        if self.pipeline:
            self.pipeline.stop()
            print("✓ 相机已停止")
        cv2.destroyAllWindows()


def main():
    """主函数"""
    # 创建检测器实例
    detector = ArucoPoseDetector(
        marker_size=0.134,
        camera_width=1280,
        camera_height=720,
        fps=6
    )

    # 运行检测
    detector.run_detection()


if __name__ == "__main__":
    main()
