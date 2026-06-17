import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import sys
from pathlib import Path
import os
import math
import time
from collections import Counter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ---------------- UI helpers: single-window ROI + ID/conf overlay ----------------

def draw_disk_id_conf(img, r0, class_name="disk"):
    """Overlay disk ID (index within current result) + confidence on img."""
    if r0 is None or getattr(r0, "boxes", None) is None:
        return img
    boxes = r0.boxes
    names = getattr(r0, "names", None)
    if names is None:
        names = {}
    try:
        cls_ids = boxes.cls.detach().cpu().numpy().astype(int)
        confs = boxes.conf.detach().cpu().numpy()
        xyxy = boxes.xyxy.detach().cpu().numpy().astype(int)
    except Exception:
        return img

    did = 0
    for i in range(len(xyxy)):
        cls_name = names.get(int(cls_ids[i]), str(int(cls_ids[i])))
        if cls_name != class_name:
            continue
        x1, y1, x2, y2 = map(int, xyxy[i])
        conf = float(confs[i])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f"ID:{did} {cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        y_text = max(th + 6, y1)
        cv2.rectangle(img, (x1, y_text - th - 6), (x1 + tw + 6, y_text), (0, 0, 0), -1)
        cv2.putText(img, label, (x1 + 3, y_text - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        did += 1
    return img


# ====================== ROI FIX PART ======================

def select_roi_in_existing_window(image, window_name, initial_roi=None):
    """
    在【原窗口】里直接拖拽选择ROI（使用OpenCV原生 selectROI，交互最稳定）
    - Enter/Space: 确认
    - ESC / 取消: 返回 None
    """
    cv2.imshow(window_name, image)
    cv2.waitKey(1)

    print("\n🖱️ ROI选择（原窗口）：拖拽选择后按 Enter/Space 确认；按 ESC 取消")
    roi = cv2.selectROI(window_name, image, showCrosshair=True, fromCenter=False)

    x, y, w, h = map(int, roi)
    if w <= 0 or h <= 0:
        print("❌ 取消ROI选择，保持原ROI不变")
        return None

    print(f"✅ ROI已保存: {(x, y, w, h)}")
    return (x, y, w, h)


# ====================== ROI FIX PART END ======================


class ROISelector:
    """Simple ROI drag selector."""

    def __init__(self, window_name="ROI Selector"):
        self.window_name = window_name
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.roi = (0, 0, 0, 0)
        self.default_roi = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            self.roi = (x, y, 0, 0)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.roi = (self.ix, self.iy, x - self.ix, y - self.iy)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1, y1 = self.ix, self.iy
            x2, y2 = x, y
            x0 = min(x1, x2)
            y0 = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            if w < 50 or h < 50:
                if self.default_roi is not None:
                    self.roi = self.default_roi
                else:
                    H, W = param.shape[:2] if isinstance(param, np.ndarray) else (720, 1280)
                    self.roi = (W // 4, H // 4, W // 2, H // 2)
            else:
                self.roi = (x0, y0, w, h)

    def select_roi(self, image):
        if self.default_roi is None:
            H, W = image.shape[:2]
            self.default_roi = (W // 4, H // 4, W // 2, H // 2)
            self.roi = self.default_roi

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback, image)
        print("\n🖱️ 拖拽选择ROI:  s保存 | r重置 | q退出并用默认")

        while True:
            disp = image.copy()
            x, y, w, h = self.roi
            if w > 0 and h > 0:
                cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(disp, f"ROI: x={x}, y={y}, w={w}, h={h}",
                            (10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(disp, "Drag ROI | s:save r:reset q:quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow(self.window_name, disp)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("s"):
                print(f"✅ ROI已保存: {self.roi}")
                break
            if k == ord("r"):
                self.roi = self.default_roi
                print("🔄 ROI已重置")
            if k == ord("q"):
                print("❌ 取消ROI选择，使用默认")
                self.roi = self.default_roi
                break

        cv2.destroyWindow(self.window_name)
        return self.roi


def euler_to_rotation_matrix(rx, ry, rz):
    rx, ry, rz = map(math.radians, (rx, ry, rz))
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(rx), -math.sin(rx)],
                   [0, math.sin(rx), math.cos(rx)]], dtype=np.float32)
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)],
                   [0, 1, 0],
                   [-math.sin(ry), 0, math.cos(ry)]], dtype=np.float32)
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0],
                   [math.sin(rz), math.cos(rz), 0],
                   [0, 0, 1]], dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)


def pose_to_transform_matrix(x, y, z, rx, ry, rz):
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = euler_to_rotation_matrix(rx, ry, rz)
    T[:3, 3] = [x, y, z]
    return T


def transform_matrix_to_pose(T):
    x, y, z = T[:3, 3]
    R = T[:3, :3]
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0.0
    return (float(x), float(y), float(z),
            float(np.degrees(rx)), float(np.degrees(ry)), float(np.degrees(rz)))


def transform_pose_cam_to_base(pose_cam, T_base_cam):
    x, y, z, rx, ry, rz = pose_cam
    T_cam_obj = pose_to_transform_matrix(x, y, z, rx, ry, rz)
    T_base_obj = T_base_cam @ T_cam_obj
    return transform_matrix_to_pose(T_base_obj)


def apply_tcp_tool_compensation(pose_base, tool_offset_xyz, enabled=True):
    """按当前姿态方向叠加 TCP 工具 XYZ 偏移；enabled=False 时原样返回。"""
    if not enabled:
        return tuple(float(v) for v in pose_base)
    x, y, z, rx, ry, rz = pose_base
    R = euler_to_rotation_matrix(rx, ry, rz).astype(np.float32)
    offset = np.array([float(tool_offset_xyz[0]), float(tool_offset_xyz[1]), float(tool_offset_xyz[2])], dtype=np.float32)
    delta = R @ offset
    return (float(x + delta[0]), float(y + delta[1]), float(z + delta[2]), float(rx), float(ry), float(rz))


def apply_manual_compensation(pose_base, manual_comp):
    """手动补偿为最终下发位姿的六轴逐项叠加。"""
    return tuple(float(a) + float(b) for a, b in zip(pose_base, manual_comp))


class RealSenseSegmentation:
    def __init__(self, model_path, width=1280, height=720, fps=6):
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.color)
        self.intrinsics = None

        print(f"🔄 加载模型: {model_path}")
        self.model = YOLO(model_path)
        self.last_r0 = None  # last YOLO result for overlay
        print("✅ 模型加载成功")

        # ROI
        self.roi_selector = ROISelector("ROI选择器")
        self.roi_enabled = True
        self.current_roi = (width // 4, height // 4, width // 2, height // 2)
        self.roi_mask = None
        self.update_roi_mask()

        # Hand-eye
        self.T_base_cam = np.eye(4, dtype=np.float32)
        self.load_calibration_matrix()

        # Table normal (camera frame)
        self.table_normal_cam = None

        # Plane fit params
        self.plane_ransac_iters = 200
        self.plane_dist_thresh_mm = 3.0
        self.plane_min_inlier_ratio = 0.65
        self.plane_max_rms_mm = 2.0
        # ---------- 后续需要改变角度大小
        # Decision thresholds (B)
        self.tilt_ok_deg = 30.0
        self.tilt_recheck_deg = 32.0
        self.tilt_vote_pass_deg = 31.0

        # GUI 运行补偿：TCP 工具补偿先作用，手动补偿最后叠加；会一直生效直到下一次修改
        self.manual_compensation = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.tool_comp_enabled = True
        self.tool_offset_xyz = [0.0, 0.0, 262.0]

    def set_manual_compensation(self, manual_comp):
        if manual_comp is None:
            manual_comp = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        if len(manual_comp) != 6:
            raise ValueError("manual_compensation 必须为 6 个数：X,Y,Z,Rx,Ry,Rz")
        self.manual_compensation = tuple(float(v) for v in manual_comp)

    def set_tool_compensation(self, enabled=True, offset_xyz=None):
        self.tool_comp_enabled = bool(enabled)
        if offset_xyz is not None:
            if len(offset_xyz) != 3:
                raise ValueError("tool_offset_xyz 必须为 3 个数：X,Y,Z")
            self.tool_offset_xyz = [float(v) for v in offset_xyz]

    # ---------- ROI helpers ----------
    def update_roi_mask(self):
        x, y, w, h = self.current_roi
        self.roi_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.rectangle(self.roi_mask, (x, y), (x + w, y + h), 255, -1)

    def set_roi(self, roi):
        self.current_roi = roi
        self.update_roi_mask()

    def toggle_roi(self, enabled=None):
        self.roi_enabled = (not self.roi_enabled) if enabled is None else bool(enabled)
        print(f"🔄 ROI过滤: {'启用' if self.roi_enabled else '禁用'}")

    def apply_roi_mask_to_image(self, image):
        if not self.roi_enabled:
            return image.copy()
        h, w = image.shape[:2]
        if self.roi_mask is None or self.roi_mask.shape != (h, w):
            self.roi_mask = np.zeros((h, w), dtype=np.uint8)
            x, y, rw, rh = self.current_roi
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            rw = max(1, min(rw, w - x))
            rh = max(1, min(rh, h - y))
            cv2.rectangle(self.roi_mask, (x, y), (x + rw, y + rh), 255, -1)
        return cv2.bitwise_and(image, image, mask=self.roi_mask)

    # ---------- camera ----------
    def start_camera(self):
        try:
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            print("🔄 启动RealSense...")
            profile = self.pipeline.start(self.config)
            color_stream = profile.get_stream(rs.stream.color)
            self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            print("✅ RealSense已启动")
            return True
        except Exception as e:
            print(f"❌ 启动失败: {e}")
            return False

    def stop_camera(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("✅ 相机已关闭")

    def capture_frames(self):
        try:
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            cf = aligned.get_color_frame()
            df = aligned.get_depth_frame()
            if not cf or not df:
                return None, None, None
            color = np.asanyarray(cf.get_data())
            depth = np.asanyarray(df.get_data())
            depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
            if self.roi_enabled:
                x, y, w, h = self.current_roi
                cv2.rectangle(color, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return color, depth, depth_vis
        except Exception as e:
            print(f"❌ 取帧失败: {e}")
            return None, None, None

    # ---------- calibration ----------
    def load_calibration_matrix(self):
        try:
            f = Path(__file__).parent / "T_base_cam.npy"
            if f.exists():
                self.T_base_cam = np.load(str(f)).astype(np.float32)
                print("✅ 加载手眼标定矩阵 T_base_cam.npy")
            else:
                print("⚠️ 未找到 T_base_cam.npy（将使用单位阵）")
        except Exception as e:
            print(f"⚠️ 加载标定矩阵失败: {e}")

    def _depth_pixels_to_points_cam(self, u, v, z_mm):
        fx, fy = self.intrinsics.fx, self.intrinsics.fy
        ppx, ppy = self.intrinsics.ppx, self.intrinsics.ppy
        x = (u - ppx) * z_mm / fx
        y = (v - ppy) * z_mm / fy
        return np.column_stack((x, y, z_mm)).astype(np.float32)

    @staticmethod
    def _fit_plane_svd(points):
        centroid = np.mean(points, axis=0)
        A = points - centroid
        _, _, vt = np.linalg.svd(A, full_matrices=False)
        n = vt[-1].astype(np.float32)
        n = n / (np.linalg.norm(n) + 1e-12)
        d = -float(np.dot(n, centroid))
        return n, d

    def _fit_plane_ransac(self, points):
        n_pts = points.shape[0]
        if n_pts < 100:
            return None
        rng = np.random.default_rng()
        best_inliers = None
        best_count = 0

        for _ in range(self.plane_ransac_iters):
            idx = rng.choice(n_pts, size=3, replace=False)
            p1, p2, p3 = points[idx]
            n = np.cross(p2 - p1, p3 - p1)
            norm = np.linalg.norm(n)
            if norm < 1e-6:
                continue
            n = (n / norm).astype(np.float32)
            d = -float(np.dot(n, p1))
            dist = np.abs(points @ n + d)
            inliers = dist < self.plane_dist_thresh_mm
            cnt = int(np.sum(inliers))
            if cnt > best_count:
                best_count = cnt
                best_inliers = inliers

        if best_inliers is None or best_count < 100:
            return None

        inlier_points = points[best_inliers]
        n_ref, d_ref = self._fit_plane_svd(inlier_points)
        dist = np.abs(inlier_points @ n_ref + d_ref)
        rms = float(np.sqrt(np.mean(dist ** 2))) if dist.size else float("inf")
        inlier_ratio = float(best_count / n_pts)
        return {"normal": n_ref, "d": d_ref, "inlier_ratio": inlier_ratio, "rms": rms}

    @staticmethod
    def _angle_between_normals_deg(n1, n2):
        n1 = n1 / (np.linalg.norm(n1) + 1e-12)
        n2 = n2 / (np.linalg.norm(n2) + 1e-12)
        c = float(np.clip(abs(np.dot(n1, n2)), -1.0, 1.0))
        return float(np.degrees(np.arccos(c)))

    def calibrate_table_normal(self, depth_image):
        if self.intrinsics is None:
            print("❌ 内参未就绪")
            return False

        if self.roi_enabled and self.roi_mask is not None:
            ys, xs = np.where(self.roi_mask > 0)
        else:
            ys, xs = np.where(np.ones_like(depth_image, dtype=np.uint8) > 0)

        z = depth_image[ys, xs].astype(np.float32)
        valid = (z > 300) & (z < 2000)
        xs, ys, z = xs[valid], ys[valid], z[valid]
        if z.size < 2000:
            print("❌ 有效点太少，标定失败（请增大ROI或靠近）")
            return False

        if z.size > 8000:
            sel = np.random.choice(z.size, size=8000, replace=False)
            xs, ys, z = xs[sel], ys[sel], z[sel]

        pts = self._depth_pixels_to_points_cam(xs.astype(np.float32), ys.astype(np.float32), z)
        plane = self._fit_plane_ransac(pts)
        if plane is None:
            print("❌ 台面平面拟合失败")
            return False

        self.table_normal_cam = plane["normal"]
        if self.table_normal_cam[2] < 0:
            self.table_normal_cam = -self.table_normal_cam
        print(f"✅ 参考平面标定完成 | inlier={plane['inlier_ratio']:.3f} rms={plane['rms']:.2f}mm")
        return True

    # ---------- decision (B) ----------
    def decision_by_tilt(self, tilt_deg, reliable):
        if tilt_deg > self.tilt_recheck_deg:
            return "STOP", f"θ={tilt_deg:.2f}° > {self.tilt_recheck_deg:.1f}°"
        if tilt_deg > self.tilt_ok_deg:
            return "RECHECK", (f"{self.tilt_ok_deg:.1f}° < θ={tilt_deg:.2f}° ≤ {self.tilt_recheck_deg:.1f}°，"
                               f"建议再测2~3帧，多数θ≤{self.tilt_vote_pass_deg:.1f}°则放行")
        if reliable:
            return "OK", f"θ={tilt_deg:.2f}° ≤ {self.tilt_ok_deg:.1f}° 且 reliable=True"
        return "RECHECK", (f"θ={tilt_deg:.2f}° ≤ {self.tilt_ok_deg:.1f}° 但 reliable=False，"
                           f"建议再测2~3帧，多数θ≤{self.tilt_vote_pass_deg:.1f}°则放行")

    # ---------- main compute ----------
    def process_pose_and_print(self, result, depth_image, quiet=False, external_final=None):
        def _p(*args, **kwargs):
            if not quiet:
                print(*args, **kwargs)

        if result.masks is None:
            _p("⚠️ 未检测到目标")
            return

        masks = result.masks.data.cpu().numpy()
        targets = []

        for i, mask in enumerate(masks):
            mask_bin = (mask > 0.5).astype(np.uint8)
            mask_bin = cv2.erode(mask_bin, np.ones((7, 7), np.uint8), iterations=1)

            v, u = np.where(mask_bin > 0)
            if u.size < 200:
                continue

            z = depth_image[v, u].astype(np.float32)
            valid = (z > 300) & (z < 1500)
            u = u[valid].astype(np.float32)
            v = v[valid].astype(np.float32)
            z = z[valid]
            if z.size < 200:
                continue

            med = np.median(z)
            mad = np.median(np.abs(z - med)) + 1e-6
            inl = np.abs(z - med) < 3.0 * mad
            u, v, z = u[inl], v[inl], z[inl]
            if z.size < 200:
                continue

            # 1. 计算3D点云，用于拟合平面 (倾斜/Tilt)
            pts = self._depth_pixels_to_points_cam(u, v, z)
            plane = self._fit_plane_ransac(pts)
            if plane is None:
                continue

            n = plane["normal"].astype(np.float32)
            if n[2] < 0:
                n = -n

            # ======================== 修改部分开始 ========================
            # 1. 找轮廓 (使用 mask_bin, 它就是歪的分割区域)
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            largest_contour = max(contours, key=cv2.contourArea)

            # 2. 计算 2D 最小外接矩形
            rect = cv2.minAreaRect(largest_contour)
            (cx_2d, cy_2d), (w_2d, h_2d), angle_2d = rect

            # ====== 【新增规则 1】长宽比保护 ======
            long_side = max(w_2d, h_2d)
            short_side = min(w_2d, h_2d)
            # 阈值设为 1.15，过滤掉近似正方形的噪点/被遮挡物体
            if short_side > 0 and (long_side / short_side) < 1.15:
                # 认为是异常形状，直接跳过
                continue
                # ====================================

            # 3. 角度对齐到长边 (解决 90度 翻转问题)
            if w_2d < h_2d:
                angle_2d = angle_2d + 90

            # ====== 【新增规则 2】角度归一化 (-90 ~ +90) ======
            # 强制将角度限制在 -90 到 90 度之间，防止 10° 和 190° 的跳变
            while angle_2d > 90:
                angle_2d -= 180
            while angle_2d < -90:
                angle_2d += 180
            # ===============================================

            # 4. 构建旋转矩阵 R
            # Z轴: 依然使用 RANSAC 算出的 n (确保吸盘贴合平面)
            z_axis = n / (np.linalg.norm(n) + 1e-12)

            # X轴: 根据 2D 角度构造方向
            theta_rad = math.radians(angle_2d)
            x_temp = np.array([math.cos(theta_rad), math.sin(theta_rad), 0.0], dtype=np.float32)

            # Y轴 = Z × X_temp (正交化)
            y_axis = np.cross(z_axis, x_temp)
            norm_y = np.linalg.norm(y_axis)
            if norm_y < 1e-6:
                y_axis = np.array([0, 1, 0], dtype=np.float32)
            else:
                y_axis /= norm_y

            # X轴 = Y × Z (重新计算 X 以保证完全正交)
            x_axis = np.cross(y_axis, z_axis)
            x_axis /= (np.linalg.norm(x_axis) + 1e-12)

            # 组合矩阵
            R = np.column_stack((x_axis, y_axis, z_axis))

            # 5. 提取欧拉角
            sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
            if sy > 1e-6:
                rx = np.arctan2(R[2, 1], R[2, 2])
                ry = np.arctan2(-R[2, 0], sy)
                rz = np.arctan2(R[1, 0], R[0, 0])
            else:
                rx = np.arctan2(-R[1, 2], R[1, 1])
                ry = np.arctan2(-R[2, 0], sy)
                rz = 0.0

            # ====== XY用新方案（2D中心反投影），Z保持老方案（点云mean的Z）======
            zc = float(np.mean(pts[:, 2]))
            center = self._depth_pixels_to_points_cam(
                np.array([cx_2d], dtype=np.float32),
                np.array([cy_2d], dtype=np.float32),
                np.array([zc], dtype=np.float32)
            )[0]
            # center[2] 已经是 zc（老版本的Z）

            pose_cam = (float(center[0]), float(center[1]), float(center[2]),
                        float(np.degrees(rx)), float(np.degrees(ry)), float(np.degrees(rz)))

            # ======================== 修改部分结束 ========================

            reliable = (plane["inlier_ratio"] >= self.plane_min_inlier_ratio) and (
                    plane["rms"] <= self.plane_max_rms_mm)

            pose_base = transform_pose_cam_to_base(pose_cam, self.T_base_cam)
            pose_base_tcp = apply_tcp_tool_compensation(
                pose_base, self.tool_offset_xyz, self.tool_comp_enabled
            )
            pose_base_final = apply_manual_compensation(pose_base_tcp, self.manual_compensation)
            height_base = pose_base_final[2]

            if self.table_normal_cam is None:
                n_ref = np.array([0, 0, 1], dtype=np.float32)
            else:
                n_ref = self.table_normal_cam

            tilt_deg = self._angle_between_normals_deg(n, n_ref)
            decision, detail = self.decision_by_tilt(tilt_deg, reliable)

            targets.append({
                "idx": i,
                "tilt": tilt_deg,
                "reliable": reliable,
                "decision": decision,
                "detail": detail,
                "pose_base": pose_base,
                "pose_base_tcp": pose_base_tcp,
                "pose_base_final": pose_base_final,
                "height_base": height_base
            })

        if not targets:
            _p("⚠️ 没有可用目标（点云不足或平面拟合失败）")
            return

        targets.sort(key=lambda t: t["height_base"], reverse=True)

        _p(f"\n✅ 检测到 {len(targets)} 个目标")
        for rank, t in enumerate(targets, start=1):
            icon = "✅" if t["decision"] == "OK" else ("🟡" if t["decision"] == "RECHECK" else "⛔")
            label = "可抓取" if t["decision"] == "OK" else ("需复核" if t["decision"] == "RECHECK" else "不可抓取")
            x, y, z, rx, ry, rz = t["pose_base_final"]
            _p(f"🎯 目标{rank} (ID:{t['idx']}) | θ={t['tilt']:.2f}° | reliable={t['reliable']} | 结论: {icon} {label}")
            _p(f"   💡 {t['detail']}")
            _p(f"   Final XYZ(mm): ({x:.1f}, {y:.1f}, {z:.1f})")
            _p(f"   Final RxRyRz(°): ({rx:.1f}, {ry:.1f}, {rz:.1f})")

        ok = next((t for t in targets if t["decision"] == "OK"), None)
        rc = next((t for t in targets if t["decision"] == "RECHECK"), None)

        final_status = "STOP"
        final_target = None
        if ok is not None:
            final_status = "OK"
            final_target = ok
        elif rc is not None:
            final_status = "RECHECK"
            final_target = rc

        if external_final is not None:
            final_status = external_final.get("status", final_status)
            final_target = external_final.get("chosen", final_target)
            theta_seq = external_final.get("theta_seq", None)
            majority_id = external_final.get("majority_id", None)
        else:
            theta_seq = None
            majority_id = None

        if not quiet:
            if final_status == "OK" and final_target is not None:
                _p(f"\n🏁 最终建议: ✅ 可抓取（3帧多数 θ≤{self.tilt_vote_pass_deg:.1f}°） | 目标ID(多数票): {final_target['idx']}")
            elif final_status == "RECHECK" and final_target is not None:
                _p(f"\n🏁 最终建议: 🟡 需复核（3帧未达多数通过） | 建议再测/重拍/调整ROI | 候选目标ID: {final_target['idx']}")
            else:
                _p(f"\n🏁 最终建议: ⛔ 不可抓取（3帧中至少1次判定 θ>{self.tilt_recheck_deg:.1f}° 或不可抓）")

            if theta_seq is not None:
                _p(f"θ序列: {[round(x, 2) for x in theta_seq]}")
            if majority_id is not None:
                _p(f"目标ID(多数票): {majority_id}")

        return {"status": final_status, "chosen": final_target, "targets": targets}


def main():
    current_dir = Path(__file__).parent
    model_path = (current_dir / "weights" / "best416.pt").resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"权重文件不存在：{model_path}")

    rs_seg = RealSenseSegmentation(str(model_path))
    if not rs_seg.start_camera():
        sys.exit(1)

    print("\n⌨️ 热键：q退出 | p计算姿态 | c标定参考平面 | r选ROI | t切ROI开关")

    try:
        while True:
            color, depth, _ = rs_seg.capture_frames()
            if color is None:
                continue

            disp = color.copy()
            disp = draw_disk_id_conf(disp, rs_seg.last_r0, class_name="disk")
            cv2.imshow("RealSense 分割检测", disp)
            k = cv2.waitKey(1) & 0xFF

            if k == ord("q"):
                break

            elif k == ord("t"):
                rs_seg.toggle_roi()

            elif k == ord("r"):
                roi_new = select_roi_in_existing_window(color, "RealSense 分割检测", initial_roi=rs_seg.current_roi)
                if roi_new is not None:
                    rs_seg.set_roi(roi_new)

            elif k == ord("c"):
                print("\n🧭 标定参考平面：请确保ROI内是空托盘/台面平面")
                rs_seg.calibrate_table_normal(depth)

            elif k == ord("p"):
                try:
                    print("\n📸 拍照完成，正在处理（自动连测3次）...")

                    N = 3
                    interval = 0.12
                    frame_store = []
                    theta_seq = []
                    id_seq = []
                    status_seq = []

                    for i in range(N):
                        color_i, depth_i, _ = rs_seg.capture_frames()
                        if color_i is None or depth_i is None:
                            continue

                        color_det_i = rs_seg.apply_roi_mask_to_image(color_i)
                        depth_pose_i = rs_seg.apply_roi_mask_to_image(depth_i)

                        results_i = rs_seg.model(color_det_i, conf=0.8, verbose=False, retina_masks=True)
                        if (not results_i) or (len(results_i) == 0) or (results_i[0].masks is None):
                            status_seq.append("NO_TARGET")
                            frame_store.append({"r0": None, "depth_pose": depth_pose_i, "res": None})
                            time.sleep(interval)
                            continue

                        r0_i = results_i[0]
                        rs_seg.last_r0 = r0_i
                        res_i = rs_seg.process_pose_and_print(r0_i, depth_pose_i, quiet=True)

                        status_seq.append(res_i.get("status", "STOP") if res_i else "STOP")
                        chosen_i = res_i.get("chosen", None) if res_i else None
                        if chosen_i is not None:
                            theta_seq.append(float(chosen_i.get("tilt", 999.0)))
                            id_seq.append(int(chosen_i.get("idx", -1)))

                        frame_store.append({"r0": r0_i, "depth_pose": depth_pose_i, "res": res_i})
                        time.sleep(interval)

                    if len(frame_store) == 0 or all(fs["r0"] is None for fs in frame_store):
                        print("\n🏁 最终建议: 🟡 需复核（未检测到目标/有效帧不足）")
                        continue

                    if any(s == "STOP" for s in status_seq):
                        final_status = "STOP"
                    else:
                        pass_cnt = sum(1 for th in theta_seq if th <= rs_seg.tilt_vote_pass_deg)
                        final_status = "OK" if pass_cnt >= 2 else "RECHECK"

                    majority_id = None
                    if len(id_seq) > 0:
                        majority_id = Counter(id_seq).most_common(1)[0][0]

                    chosen_frame = None
                    for fs in frame_store:
                        if fs["res"] and fs["res"].get("chosen") is not None and majority_id is not None:
                            if int(fs["res"]["chosen"]["idx"]) == int(majority_id):
                                chosen_frame = fs
                                break
                    if chosen_frame is None:
                        chosen_frame = next(
                            (fs for fs in frame_store if fs["r0"] is not None and fs["res"] is not None),
                            frame_store[0])

                    chosen_target = None
                    if chosen_frame.get("res") and chosen_frame["res"].get("targets"):
                        if majority_id is not None:
                            chosen_target = next(
                                (t for t in chosen_frame["res"]["targets"] if int(t["idx"]) == int(majority_id)), None)
                        if chosen_target is None:
                            chosen_target = chosen_frame["res"].get("chosen", None)

                    external_final = {
                        "status": final_status,
                        "chosen": chosen_target,
                        "theta_seq": theta_seq,
                        "majority_id": majority_id,
                    }

                    rs_seg.process_pose_and_print(chosen_frame["r0"], chosen_frame["depth_pose"], quiet=False,
                                                  external_final=external_final)

                except Exception:
                    import traceback
                    print("❌ 按p处理发生异常（程序继续运行）：")
                    traceback.print_exc()
                    continue

    finally:
        rs_seg.stop_camera()


if __name__ == "__main__":
    main()