import sys
import os
import asyncio
import cv2
import numpy as np
import json
import time
import shutil
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QPushButton, QVBoxLayout, QHBoxLayout, QStackedLayout,
                             QMessageBox, QTextEdit, QGroupBox, QLineEdit, QGridLayout,
                             QDialog, QDialogButtonBox, QPlainTextEdit, QSizePolicy, QScrollArea,
                             QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRect
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QPen, QColor, QDoubleValidator

# ================== 导入核心模块 ==================
try:
    from detect import RealSenseSegmentation, draw_disk_id_conf
    from json_payloads import QueryPayloadFactory
    from apritag_detect import ArucoPoseDetector
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# ================== 全局核心配置 (完全对齐 test.py) ==================
ROBOT_IP = "192.168.4.4"
ROBOT_PORT = 9760
FEEDBACK_DELAY = 2.0  # 反馈后强制等待时间
TIMEOUT = 15.0  # 坐标指令响应超时时间
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "weights" / "seg-s.pt"

# 文件路径定义：全部固定到本脚本所在目录，避免从不同工作目录启动时跑到项目外面
FILE_CAM = BASE_DIR / "apriltag_poses.txt"
FILE_ROBOT = BASE_DIR / "robot_base_poses.txt"
FILE_CAM_ORIGIN = BASE_DIR / "apriltag_poses_original.txt"
FILE_ROBOT_ORIGIN = BASE_DIR / "robot_base_poses_original.txt"
FILE_CAM_TEMP = BASE_DIR / "apriltag_poses.temp"
FILE_ROBOT_TEMP = BASE_DIR / "robot_base_poses.temp"
FILE_T_BASE_CAM_NPY = BASE_DIR / "T_base_cam.npy"
FILE_T_BASE_CAM_TXT = BASE_DIR / "T_base_cam.txt"
MAX_GUI_LOG_LINES = 1000  # GUI日志最多保留行数，防止长时间运行内存持续增长
FILE_RUNTIME_CONFIG = BASE_DIR / "runtime_config.json"
DEFAULT_Z_MIN_LIMIT = 381.0
DEFAULT_SOFT_LIMIT_ENABLED = True


# ================== 辅助函数 ==================
def get_next_index(filename):
    if not os.path.exists(filename): return 1
    count = 0
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if ',' in line and not line.strip().startswith(('//', '#')):
                count += 1
    return count + 1


def ensure_original_backups():
    if not os.path.exists(FILE_CAM_ORIGIN) and os.path.exists(FILE_CAM):
        shutil.copy(FILE_CAM, FILE_CAM_ORIGIN)
    if not os.path.exists(FILE_ROBOT_ORIGIN) and os.path.exists(FILE_ROBOT):
        shutil.copy(FILE_ROBOT, FILE_ROBOT_ORIGIN)


def comp_to_text(comp):
    return (
        f"X={comp[0]:+.3f}, Y={comp[1]:+.3f}, Z={comp[2]:+.3f}, "
        f"Rx={comp[3]:+.3f}, Ry={comp[4]:+.3f}, Rz={comp[5]:+.3f}"
    )


# ================== 弹窗编辑器 ==================
class FileEditorDialog(QDialog):
    def __init__(self, title, file_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(700, 500)
        self.file_path = file_path
        layout = QVBoxLayout()
        lbl_warn = QLabel("⚠️ 警告：每行只保留 6 个数：X,Y,Z,Rx,Ry,Rz；视觉点和机械臂点按行号一一对应。")
        lbl_warn.setStyleSheet(
            "color: red; font-weight: bold; font-size: 14px; border: 1px solid red; padding: 10px; background: #FFF0F0;")
        layout.addWidget(lbl_warn)
        self.text_edit = QPlainTextEdit()
        self.text_edit.setStyleSheet("font-family: Consolas; font-size: 14px;")
        layout.addWidget(self.text_edit)
        self.load_file()
        btn_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.save_file)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)
        self.setLayout(layout)

    def load_file(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.text_edit.setPlainText(f.read())
        else:
            self.text_edit.setPlainText("")

    def save_file(self):
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                f.write(self.text_edit.toPlainText())
            QMessageBox.information(self, "保存", "文件已更新！")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {e}")


# ================== ROI 编辑页面 ==================
class ROIEditPage(QWidget):
    roi_confirmed = pyqtSignal(tuple)
    cancelled = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.lbl_instruction = QLabel("请按住鼠标左键，在画面上拖动以更新检测范围")
        self.lbl_instruction.setStyleSheet(
            "font-size: 24px; color: yellow; font-weight: bold; background: #333; padding: 10px;")
        self.lbl_instruction.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.lbl_instruction)
        self.canvas = ROICanvas()
        self.canvas.mouse_released_signal.connect(self.on_selection_finished)
        self.layout.addWidget(self.canvas, stretch=1)
        self.btn_layout = QHBoxLayout()
        self.btn_cancel = QPushButton("❌ 不更新，返回检测")
        self.btn_cancel.setStyleSheet(
            "background-color: #7f8c8d; color: white; font-size: 18px; padding: 15px; border-radius: 5px;")
        self.btn_cancel.clicked.connect(self.on_cancel)
        self.btn_redraw = QPushButton("🔄 重新拖动")
        self.btn_redraw.setStyleSheet(
            "background-color: #e67e22; color: white; font-size: 18px; padding: 15px; border-radius: 5px;")
        self.btn_redraw.clicked.connect(self.on_redraw)
        self.btn_redraw.hide()
        self.btn_confirm = QPushButton("✅ 确认更新，返回检测")
        self.btn_confirm.setStyleSheet(
            "background-color: #27ae60; color: white; font-size: 18px; padding: 15px; border-radius: 5px;")
        self.btn_confirm.clicked.connect(self.on_confirm)
        self.btn_confirm.hide()
        self.btn_layout.addWidget(self.btn_cancel)
        self.btn_layout.addStretch()
        self.btn_layout.addWidget(self.btn_redraw)
        self.btn_layout.addWidget(self.btn_confirm)
        self.layout.addLayout(self.btn_layout)

    def set_image(self, cv_img):
        self.canvas.set_image(cv_img)
        self.on_redraw()

    def on_selection_finished(self):
        self.lbl_instruction.setText("已选择区域，请确认或重画")
        self.lbl_instruction.setStyleSheet(
            "font-size: 24px; color: #00ff00; font-weight: bold; background: #333; padding: 10px;")
        self.btn_redraw.show()
        self.btn_confirm.show()

    def on_redraw(self):
        self.canvas.reset_selection()
        self.lbl_instruction.setText("请按住鼠标左键，在画面上拖动以更新检测范围")
        self.lbl_instruction.setStyleSheet(
            "font-size: 24px; color: yellow; font-weight: bold; background: #333; padding: 10px;")
        self.btn_redraw.hide()
        self.btn_confirm.hide()

    def on_confirm(self):
        roi = self.canvas.get_roi_original_scale()
        if roi:
            self.roi_confirmed.emit(roi)
        else:
            self.cancelled.emit()

    def on_cancel(self):
        self.cancelled.emit()


class ROICanvas(QLabel):
    mouse_released_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: black;")
        self.setMouseTracking(True)
        self.original_cv_img = None
        self.pixmap_item = None
        self.start_pos = None
        self.end_pos = None
        self.is_drawing = False
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0

    def set_image(self, cv_img):
        self.original_cv_img = cv_img
        h, w, c = cv_img.shape
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, c * w, QImage.Format_RGB888)
        self.pixmap_item = QPixmap.fromImage(qimg)
        self.update()

    def reset_selection(self):
        self.start_pos = None
        self.end_pos = None
        self.is_drawing = False
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_pos = event.pos()
            self.end_pos = event.pos()
            self.is_drawing = True
            self.update()

    def mouseMoveEvent(self, event):
        if self.is_drawing:
            self.end_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_drawing = False
            self.end_pos = event.pos()
            self.update()
            if self.start_pos and self.end_pos:
                if (self.start_pos - self.end_pos).manhattanLength() > 10:
                    self.mouse_released_signal.emit()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.pixmap_item is None: return
        painter = QPainter(self)
        w_label = self.width()
        h_label = self.height()
        w_img = self.pixmap_item.width()
        scaled_pixmap = self.pixmap_item.scaled(w_label, h_label, Qt.KeepAspectRatio)
        self.offset_x = (w_label - scaled_pixmap.width()) // 2
        self.offset_y = (h_label - scaled_pixmap.height()) // 2
        self.scale_factor = scaled_pixmap.width() / w_img
        painter.drawPixmap(self.offset_x, self.offset_y, scaled_pixmap)
        if self.start_pos and self.end_pos:
            pen = QPen(QColor(0, 255, 0), 2, Qt.SolidLine)
            painter.setPen(pen)
            rect = QRect(self.start_pos, self.end_pos).normalized()
            painter.drawRect(rect)

    def get_roi_original_scale(self):
        if not self.start_pos or not self.end_pos: return None
        x1_sc = min(self.start_pos.x(), self.end_pos.x())
        y1_sc = min(self.start_pos.y(), self.end_pos.y())
        x2_sc = max(self.start_pos.x(), self.end_pos.x())
        y2_sc = max(self.start_pos.y(), self.end_pos.y())
        x1 = int((x1_sc - self.offset_x) / self.scale_factor)
        y1 = int((y1_sc - self.offset_y) / self.scale_factor)
        x2 = int((x2_sc - self.offset_x) / self.scale_factor)
        y2 = int((y2_sc - self.offset_y) / self.scale_factor)
        h_orig, w_orig = self.original_cv_img.shape[:2]
        x1 = max(0, min(x1, w_orig))
        y1 = max(0, min(y1, h_orig))
        x2 = max(0, min(x2, w_orig))
        y2 = max(0, min(y2, h_orig))
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0: return None
        return (x1, y1, w, h)


# ================== 通信线程 ==================
class RobotCommWorker(QThread):
    log_signal = pyqtSignal(str)
    photo_signal = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.loop = None
        self.running = True
        self.soft_limit_enabled = DEFAULT_SOFT_LIMIT_ENABLED
        self.z_min_limit = DEFAULT_Z_MIN_LIMIT

    def set_soft_limit(self, enabled, z_min_limit):
        self.soft_limit_enabled = bool(enabled)
        self.z_min_limit = float(z_min_limit)

    def run(self):
        if os.name == 'nt': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.listen_robot())

    async def listen_robot(self):
        self.log_signal.emit(f"📡 连接中: {ROBOT_IP}:{ROBOT_PORT}")
        while self.running:
            try:
                reader, writer = await asyncio.open_connection(ROBOT_IP, ROBOT_PORT)
                self.log_signal.emit("✅ 监听连接建立成功")
                while self.running:
                    try:
                        data = await asyncio.wait_for(reader.read(4096), timeout=1.0)
                        if not data: break
                        msg = data.decode("utf-8", errors='ignore').strip()
                        try:
                            cmd = json.loads(msg)
                            if cmd.get("reqType") == "photo":
                                self.log_signal.emit(f"⚡ 收到指令: 拍照 (ID:{cmd.get('camID')})")
                                self.photo_signal.emit(int(cmd.get("camID", 0)))
                        except:
                            pass
                    except asyncio.TimeoutError:
                        continue
                    except:
                        break
                writer.close()
                await writer.wait_closed()
                self.log_signal.emit("🔌 断开重连中...")
            except:
                await asyncio.sleep(2)

    def stop(self):
        self.running = False
        self.wait()

    def send_feedback(self, cid, success):
        if self.loop: asyncio.run_coroutine_threadsafe(self._snd_fb(cid, success), self.loop)

    def send_success_sequence(self, cid, pose):
        if self.loop: asyncio.run_coroutine_threadsafe(self._snd_seq(cid, pose), self.loop)

    async def _snd_seq(self, cid, pose):
        await self._snd_fb(cid, True)
        self.log_signal.emit(f"⏳ 等待 {FEEDBACK_DELAY}s 发送坐标...")
        await asyncio.sleep(FEEDBACK_DELAY)
        await self._snd_pt(cid, pose)

    async def _snd_fb(self, cid, success):
        try:
            p = QueryPayloadFactory.create_photo_payload(cid, 1 if success else 0)
            _, w = await asyncio.open_connection(ROBOT_IP, ROBOT_PORT)
            w.write(json.dumps(p).encode())
            await w.drain()
            w.close()
            await w.wait_closed()
            self.log_signal.emit(f"📤 反馈发送成功: {'OK' if success else 'NG'}")
        except Exception as e:
            self.log_signal.emit(f"❌ 反馈错: {e}")

    async def _snd_pt(self, cid, pose):
        try:
            x, y, z, rx, ry, rz = pose

            # 对齐 test.py：Z轴安全抬升 80.0
            SAFE_HEIGHT_OFFSET = 80.0
            z_safe_send = z + SAFE_HEIGHT_OFFSET

            # Z轴软限位：可在维护面板中修改，配置会写入 runtime_config.json 持续生效
            if self.soft_limit_enabled and z_safe_send < self.z_min_limit:
                self.log_signal.emit(
                    f"🚨 触发软限位！高度 Z={z_safe_send:.1f} 低于阈值 {self.z_min_limit:.1f}，已强制覆写"
                )
                z_safe_send = self.z_min_limit

            pd = {
                "ModelID": "0",
                "X": f"{x:.3f}",
                "Y": f"{y:.3f}",
                "Z": f"{z_safe_send:.3f}",
                "U": f"{rx:.3f}",
                "V": f"{ry:.3f}",
                "Angle": f"{rz:.3f}",
                "Similarity": "0",
                "Color": "0",
                "Rel": "0"
            }
            p = QueryPayloadFactory.create_add_points_payload("1", "1", str(cid), [pd])

            r, w = await asyncio.open_connection(ROBOT_IP, ROBOT_PORT)
            w.write(json.dumps(p).encode())
            await w.drain()

            try:
                response_data = await asyncio.wait_for(r.read(4096), timeout=TIMEOUT)
                if response_data:
                    self.log_signal.emit(f"📥 机械臂响应：{response_data.decode('utf-8', errors='ignore')}")
            except asyncio.TimeoutError:
                self.log_signal.emit("⚠️ 坐标指令响应超时")

            w.close()
            await w.wait_closed()
            self.log_signal.emit(f"✅ 坐标指令发送完成 (实际发送Z={z_safe_send:.1f})")

        except Exception as e:
            self.log_signal.emit(f"❌ 发送坐标异常: {e}")


# ================== 视觉线程 ==================
class VideoThread(QThread):
    img_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.running = True
        self.mode = "DETECT"
        self.rs_cam = None
        self.aruco = None
        self._manual_comp = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._tool_comp_enabled = True
        self._tool_offset_xyz = [0.0, 0.0, 262.0]

    def run(self):
        try:
            self.rs_cam = RealSenseSegmentation(str(WEIGHTS_PATH))
            if not self.rs_cam.start_camera(): return

            self.aruco = ArucoPoseDetector()
            self.aruco.setup_aruco_detector()
            self.aruco.camera_matrix = np.array([[self.rs_cam.intrinsics.fx, 0, self.rs_cam.intrinsics.ppx],
                                                 [0, self.rs_cam.intrinsics.fy, self.rs_cam.intrinsics.ppy], [0, 0, 1]])
            self.aruco.dist_coeffs = np.array(self.rs_cam.intrinsics.coeffs)

            self.rs_cam.set_manual_compensation(self._manual_comp)
            self.rs_cam.set_tool_compensation(self._tool_comp_enabled, self._tool_offset_xyz)
        except Exception as e:
            import traceback
            print("❌ VideoThread 启动异常:", e)
            traceback.print_exc()
            return

        while self.running:
            c, d, _ = self.rs_cam.capture_frames()
            if c is None: continue

            if self.mode == "DETECT":
                if self.rs_cam.roi_enabled:
                    cv2.rectangle(c, (self.rs_cam.current_roi[0], self.rs_cam.current_roi[1]),
                                  (self.rs_cam.current_roi[0] + self.rs_cam.current_roi[2],
                                   self.rs_cam.current_roi[1] + self.rs_cam.current_roi[3]), (0, 255, 0), 2)
                c = draw_disk_id_conf(c, self.rs_cam.last_r0)

            elif self.mode == "CALIB":
                crn, ids, poses = self.aruco.detect_markers(c)
                self.aruco.draw_axis(c, crn, ids, poses)
                self.cur_poses = poses

            self.img_signal.emit(c)
            QThread.msleep(30)

        self.rs_cam.stop_camera()

    def stop(self):
        self.running = False
        self.wait()

    def set_manual_compensation(self, manual_comp):
        self._manual_comp = tuple(float(v) for v in manual_comp)
        if self.rs_cam:
            self.rs_cam.set_manual_compensation(self._manual_comp)

    def set_tool_compensation(self, enabled, offset_xyz):
        self._tool_comp_enabled = bool(enabled)
        self._tool_offset_xyz = [float(v) for v in offset_xyz]
        if self.rs_cam:
            self.rs_cam.set_tool_compensation(self._tool_comp_enabled, self._tool_offset_xyz)

    def reload_hand_eye_matrix(self):
        if self.rs_cam:
            self.rs_cam.load_calibration_matrix()

    def trigger_detection(self):
        if not self.rs_cam: return None, "无相机"

        c, d, _ = self.rs_cam.capture_frames()
        if c is None: return None, "未能获取图像"

        c_roi = self.rs_cam.apply_roi_mask_to_image(c)
        d_roi = self.rs_cam.apply_roi_mask_to_image(d)

        # 单帧识别
        res = self.rs_cam.model(c_roi, conf=0.8, verbose=False, retina_masks=True)
        if not res or len(res) == 0 or res[0].masks is None:
            return None, "未检测到硬盘"

        self.rs_cam.last_r0 = res[0]

        ret = self.rs_cam.process_pose_and_print(res[0], d_roi, quiet=True)

        if ret and ret.get("targets"):
            highest = ret["targets"][0]
            pose_final = highest.get("pose_base_final", highest.get("pose_base"))
            return pose_final, "按高度选择最高目标（已应用 TCP/手动补偿）"

        return None, "未找到有效位姿"


# ================== 视频显示控件：保持正常比例并铺满可用区域 ==================
class VideoDisplayLabel(QLabel):
    def __init__(self, text="启动相机中..."):
        super().__init__(text)
        self._pixmap_original = None
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:black; color:white; font-size:20px;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(320, 240)

    def set_video_pixmap(self, pixmap):
        self._pixmap_original = pixmap
        self._refresh_scaled_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_scaled_pixmap()

    def _refresh_scaled_pixmap(self):
        if self._pixmap_original is None or self.width() <= 0 or self.height() <= 0:
            return
        scaled = self._pixmap_original.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)

# ================= 主界面 =================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        ensure_original_backups()
        self.setWindowTitle("Robot Vision (GUI Pro)")
        self.resize(1024, 768)
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.stack = QStackedLayout()
        self.main_widget.setLayout(self.stack)

        self.calib_is_dirty = False
        self.calib_calculated = False

        self.manual_comp = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.tool_comp_enabled = True
        self.tool_offset_xyz = [0.0, 0.0, 262.0]
        self.soft_limit_enabled = DEFAULT_SOFT_LIMIT_ENABLED
        self.z_min_limit = DEFAULT_Z_MIN_LIMIT

        self.page_auto = QWidget()
        self.init_ui_auto()
        self.stack.addWidget(self.page_auto)

        self.page_calib = QWidget()
        self.init_ui_calib_pro()
        self.stack.addWidget(self.page_calib)

        self.page_roi = ROIEditPage()
        self.page_roi.roi_confirmed.connect(self.on_roi_updated)
        self.page_roi.cancelled.connect(self.on_roi_cancelled)
        self.stack.addWidget(self.page_roi)

        self.vt = VideoThread()
        self.vt.img_signal.connect(self.update_img)
        self.vt.start()

        self.ct = RobotCommWorker()
        self.ct.log_signal.connect(self.log)
        self.ct.photo_signal.connect(self.on_robot_trigger)
        self.ct.start()

        self.load_runtime_config()
        self.apply_runtime_settings_to_video_thread()

        self.log("💻 系统启动 | 已连接机械臂网络")

    def init_ui_auto(self):
        outer_layout = QVBoxLayout()

        self.lbl_v1 = VideoDisplayLabel("启动相机中...")
        outer_layout.addWidget(self.lbl_v1, stretch=1)

        maintenance_row = QHBoxLayout()
        self.btn_toggle_maintenance = QPushButton("🔧 修改")
        self.btn_toggle_maintenance.setFixedWidth(120)
        self.btn_toggle_maintenance.setStyleSheet(
            "QPushButton{font-size:16px; padding:10px; background:#444; color:white; border-radius:6px;}"
        )
        self.btn_toggle_maintenance.clicked.connect(self.toggle_maintenance_panel)
        maintenance_row.addWidget(self.btn_toggle_maintenance)
        maintenance_row.addStretch()
        outer_layout.addLayout(maintenance_row)

        self.bottom_scroll = QScrollArea()
        self.bottom_scroll.setWidgetResizable(True)
        self.bottom_scroll.setStyleSheet("QScrollArea { border: none; background: #F5F5F5; }")
        self.bottom_container = QWidget()
        self.bottom_layout = QVBoxLayout(self.bottom_container)
        self.bottom_layout.setContentsMargins(10, 10, 10, 10)
        self.bottom_layout.setSpacing(10)
        self.bottom_scroll.setWidget(self.bottom_container)
        outer_layout.addWidget(self.bottom_scroll, stretch=0)
        self.bottom_scroll.hide()

        self.txt_log = QTextEdit()
        self.txt_log.setFixedHeight(120)
        self.txt_log.setReadOnly(True)
        self.txt_log.setStyleSheet("background:#222; color:#0f0;")
        self.bottom_layout.addWidget(self.txt_log)

        btns = QHBoxLayout()
        st = "QPushButton{font-size:18px; padding:15px; background:#0078D7; color:white; border-radius:8px;}"
        b1 = QPushButton("🛠️ 手眼标定")
        b1.setStyleSheet(st)
        b1.clicked.connect(self.enter_calib_mode)
        b2 = QPushButton("📐 修改范围")
        b2.setStyleSheet(st)
        b2.clicked.connect(self.start_roi_edit)
        b3 = QPushButton("🧭 修改检测平面")
        b3.setStyleSheet(st)
        b3.clicked.connect(self.do_plane)
        b3.hide()
        btns.addWidget(b1)
        btns.addWidget(b2)
        btns.addWidget(b3)
        self.bottom_layout.addLayout(btns)

        grp_manual = QGroupBox("本次运行手动补偿（叠加在 TCP 补偿后）")
        grp_manual.setStyleSheet("QGroupBox { font-weight: bold; font-size: 15px; border: 1px solid gray; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        manual_layout = QVBoxLayout()
        grid_manual = QGridLayout()
        self.manual_inputs = {}
        validator = QDoubleValidator()
        for i, label in enumerate(["X", "Y", "Z", "Rx", "Ry", "Rz"]):
            l = QLabel(f"{label}:")
            l.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            inp = QLineEdit("0.000")
            inp.setValidator(validator)
            self.manual_inputs[label] = inp
            grid_manual.addWidget(l, i // 3, (i % 3) * 2)
            grid_manual.addWidget(inp, i // 3, (i % 3) * 2 + 1)
        manual_layout.addLayout(grid_manual)
        row_manual_btn = QHBoxLayout()
        self.btn_apply_manual = QPushButton("✅ 应用手动补偿")
        self.btn_apply_manual.clicked.connect(self.apply_manual_comp_from_ui)
        self.btn_zero_manual = QPushButton("↺ 清零手动补偿")
        self.btn_zero_manual.clicked.connect(self.clear_manual_comp)
        row_manual_btn.addWidget(self.btn_apply_manual)
        row_manual_btn.addWidget(self.btn_zero_manual)
        manual_layout.addLayout(row_manual_btn)
        self.lbl_manual_status = QLabel("当前手动补偿：X=+0.000, Y=+0.000, Z=+0.000, Rx=+0.000, Ry=+0.000, Rz=+0.000")
        self.lbl_manual_status.setWordWrap(True)
        self.lbl_manual_status.setStyleSheet("background:#F6F6F6; border:1px solid #CCC; padding:6px;")
        manual_layout.addWidget(self.lbl_manual_status)
        grp_manual.setLayout(manual_layout)
        self.bottom_layout.addWidget(grp_manual)

        grp_tool = QGroupBox("TCP 工具补偿")
        grp_tool.setStyleSheet("QGroupBox { font-weight: bold; font-size: 15px; border: 1px solid gray; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        tool_layout = QVBoxLayout()
        self.lbl_tool_enable = QLabel()
        self.lbl_tool_enable.setWordWrap(True)
        self.lbl_tool_enable.setStyleSheet("background:#F6F6F6; border:1px solid #CCC; padding:6px;")
        tool_layout.addWidget(self.lbl_tool_enable)
        grid_tool = QGridLayout()
        self.tool_inputs = {}
        validator_tool = QDoubleValidator()
        for i, label in enumerate(["X", "Y", "Z"]):
            l = QLabel(f"{label}:")
            l.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            inp = QLineEdit()
            inp.setValidator(validator_tool)
            self.tool_inputs[label] = inp
            grid_tool.addWidget(l, 0, i * 2)
            grid_tool.addWidget(inp, 0, i * 2 + 1)
        tool_layout.addLayout(grid_tool)
        row_tool_btn = QHBoxLayout()
        self.btn_tool_toggle = QPushButton("🔁 启用/禁用工具补偿")
        self.btn_tool_toggle.clicked.connect(self.toggle_tool_compensation)
        self.btn_tool_apply = QPushButton("✅ 应用工具偏移")
        self.btn_tool_apply.clicked.connect(self.apply_tool_comp_from_ui)
        row_tool_btn.addWidget(self.btn_tool_toggle)
        row_tool_btn.addWidget(self.btn_tool_apply)
        tool_layout.addLayout(row_tool_btn)
        grp_tool.setLayout(tool_layout)
        self.bottom_layout.addWidget(grp_tool)

        grp_soft = QGroupBox("Z轴软限位")
        grp_soft.setStyleSheet("QGroupBox { font-weight: bold; font-size: 15px; border: 1px solid gray; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        soft_layout = QVBoxLayout()
        self.lbl_soft_limit_status = QLabel()
        self.lbl_soft_limit_status.setWordWrap(True)
        self.lbl_soft_limit_status.setStyleSheet("background:#FFF8E8; border:2px solid #E0B45A; padding:10px; font-size:22px; font-weight:bold;")
        soft_layout.addWidget(self.lbl_soft_limit_status)
        soft_grid = QGridLayout()
        soft_grid.addWidget(QLabel("Z最小值:"), 0, 0)
        self.input_z_min_limit = QLineEdit(f"{DEFAULT_Z_MIN_LIMIT:.3f}")
        self.input_z_min_limit.setValidator(QDoubleValidator())
        soft_grid.addWidget(self.input_z_min_limit, 0, 1)
        self.btn_soft_toggle = QPushButton("🔁 启用/禁用软限位")
        self.btn_soft_toggle.clicked.connect(self.toggle_soft_limit)
        self.btn_soft_apply = QPushButton("✅ 应用软限位数值")
        self.btn_soft_apply.clicked.connect(self.apply_soft_limit_from_ui)
        soft_grid.addWidget(self.btn_soft_toggle, 1, 0)
        soft_grid.addWidget(self.btn_soft_apply, 1, 1)
        soft_layout.addLayout(soft_grid)
        grp_soft.setLayout(soft_layout)
        self.bottom_layout.addWidget(grp_soft)

        self.lbl_runtime_status = QLabel("等待初始化...")
        self.lbl_runtime_status.setWordWrap(True)
        self.lbl_runtime_status.setStyleSheet("background:#EEF5FF; border:1px solid #90B4E8; padding:8px;")
        self.bottom_layout.addWidget(self.lbl_runtime_status)

        self.bottom_layout.addStretch()
        self.page_auto.setLayout(outer_layout)

    def toggle_maintenance_panel(self):
        show = self.bottom_scroll.isHidden()
        self.bottom_scroll.setVisible(show)
        self.btn_toggle_maintenance.setText("🔧 关闭修改面板" if show else "🔧 修改")

    def init_ui_calib_pro(self):
        layout = QVBoxLayout()
        header = QHBoxLayout()
        title = QLabel("🛠️ 标定模式")
        title.setFont(QFont("Microsoft YaHei", 20, QFont.Bold))
        instruction = QLabel(
            "请将标定板放置在镜头下，等待识别成功后采集标定点，采集后，控制机械臂使夹爪中心对准标定板中心，并手动输入机械臂坐标，并保存。")
        instruction.setStyleSheet("color: #666; font-size: 14px; margin-left: 20px;")

        self.btn_reset_origin = QPushButton("🚨 恢复原始备份 (兜底)")
        self.btn_reset_origin.setStyleSheet("background-color: #C0392B; color: white; padding: 8px; font-weight: bold;")
        self.btn_reset_origin.clicked.connect(self.restore_original_backup)

        self.btn_overwrite_origin = QPushButton("💾 覆盖原始备份")
        self.btn_overwrite_origin.setStyleSheet("background-color: #16A085; color: white; padding: 8px; font-weight: bold;")
        self.btn_overwrite_origin.clicked.connect(self.overwrite_original_backup)

        btn_back = QPushButton("🔙 返回")
        btn_back.clicked.connect(self.exit_calib_mode)
        header.addWidget(title)
        header.addWidget(instruction)
        header.addStretch()
        header.addWidget(self.btn_reset_origin)
        header.addWidget(self.btn_overwrite_origin)
        header.addWidget(btn_back)
        layout.addLayout(header)

        # --- 修改部分: 标定页面同样加上滚动区域，保持等比且允许滑块 ---
        self.lbl_v2 = QLabel()
        self.lbl_v2.setStyleSheet("background: black;")
        self.lbl_v2.setAlignment(Qt.AlignCenter)

        self.scroll_v2 = QScrollArea()
        self.scroll_v2.setWidget(self.lbl_v2)
        self.scroll_v2.setWidgetResizable(True)
        self.scroll_v2.setAlignment(Qt.AlignCenter)
        self.scroll_v2.setStyleSheet("QScrollArea { border: 2px solid #555; }")

        layout.addWidget(self.scroll_v2, stretch=1)
        # -------------------------------------------------------------

        bottom_area = QHBoxLayout()

        grp_cam = QGroupBox("第一步：视觉采集")
        grp_cam.setStyleSheet(
            "QGroupBox { font-weight: bold; font-size: 16px; border: 1px solid gray; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        cam_layout = QVBoxLayout()
        self.btn_cap_cam = QPushButton("📸 采集标定点")
        self.btn_cap_cam.setStyleSheet(
            "background-color: #E67E22; color: white; font-size: 16px; padding: 10px; border-radius: 5px;")
        self.btn_cap_cam.clicked.connect(self.capture_camera_point)
        cam_layout.addWidget(self.btn_cap_cam)

        self.lbl_cam_status = QLabel("等待采集...")
        self.lbl_cam_status.setStyleSheet(
            "background: #EEE; border: 1px solid #CCC; padding: 5px; font-size: 14px; color: blue;")
        self.lbl_cam_status.setAlignment(Qt.AlignCenter)
        self.lbl_cam_status.setFixedHeight(40)
        cam_layout.addWidget(self.lbl_cam_status)

        self.btn_view_cam_file = QPushButton("📄 查看视觉数据 (apriltag_poses.txt)")
        self.btn_view_cam_file.clicked.connect(lambda: self.open_file_editor("视觉点数据", FILE_CAM))
        cam_layout.addWidget(self.btn_view_cam_file)

        # 新增查看标定结果按钮
        self.btn_view_calib_result = QPushButton("📊 查看标定结果 (T_base_cam.txt)")
        self.btn_view_calib_result.clicked.connect(self.view_calib_result)
        cam_layout.addWidget(self.btn_view_calib_result)

        cam_layout.addStretch()
        grp_cam.setLayout(cam_layout)

        grp_rob = QGroupBox("第二步：输入机械臂坐标")
        grp_rob.setStyleSheet(
            "QGroupBox { font-weight: bold; font-size: 16px; border: 1px solid gray; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        rob_layout = QVBoxLayout()
        lbl_rob_hint = QLabel("采集视觉点后，移动机械臂对准标定板中心，\n读取坐标并填入下方：")
        lbl_rob_hint.setStyleSheet("color: #333; font-size: 12px;")
        rob_layout.addWidget(lbl_rob_hint)

        grid_input = QGridLayout()
        self.inputs = {}
        labels = ['X', 'Y', 'Z', 'Rx', 'Ry', 'Rz']
        for i, label in enumerate(labels):
            l = QLabel(f"{label}:")
            l.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            inp = QLineEdit()
            inp.setPlaceholderText("0.000")
            inp.setValidator(QDoubleValidator())
            self.inputs[label] = inp
            grid_input.addWidget(l, i // 3, (i % 3) * 2)
            grid_input.addWidget(inp, i // 3, (i % 3) * 2 + 1)
        rob_layout.addLayout(grid_input)

        self.btn_save_rob = QPushButton("💾 保存机械臂坐标")
        self.btn_save_rob.setStyleSheet(
            "background-color: #2980B9; color: white; font-size: 16px; padding: 10px; border-radius: 5px;")
        self.btn_save_rob.clicked.connect(self.save_robot_point)
        rob_layout.addWidget(self.btn_save_rob)

        self.lbl_rob_status = QLabel("等待输入...")
        self.lbl_rob_status.setStyleSheet(
            "background: #EEE; border: 1px solid #CCC; padding: 5px; font-size: 14px; color: blue;")
        self.lbl_rob_status.setAlignment(Qt.AlignCenter)
        self.lbl_rob_status.setFixedHeight(40)
        rob_layout.addWidget(self.lbl_rob_status)

        self.btn_view_rob_file = QPushButton("📄 查看机械臂数据 (robot_base_poses.txt)")
        self.btn_view_rob_file.clicked.connect(lambda: self.open_file_editor("机械臂点数据", FILE_ROBOT))
        rob_layout.addWidget(self.btn_view_rob_file)
        grp_rob.setLayout(rob_layout)

        bottom_area.addWidget(grp_cam, stretch=4)
        bottom_area.addWidget(grp_rob, stretch=6)

        layout.addLayout(bottom_area)

        self.btn_calc_final = QPushButton("🧮 采集完成，开始计算标定矩阵 (T_base_cam)")
        self.btn_calc_final.setStyleSheet(
            "background-color: #27AE60; color: white; font-weight: bold; font-size: 18px; padding: 15px;")
        self.btn_calc_final.clicked.connect(self.calc)
        layout.addWidget(self.btn_calc_final)
        self.page_calib.setLayout(layout)

    # ================= 运行补偿参数 =================
    def load_runtime_config(self):
        if os.path.exists(FILE_RUNTIME_CONFIG):
            try:
                with open(FILE_RUNTIME_CONFIG, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                self.manual_comp = tuple(float(v) for v in cfg.get("manual_comp", [0, 0, 0, 0, 0, 0]))
                self.tool_comp_enabled = bool(cfg.get("tool_comp_enabled", True))
                self.tool_offset_xyz = [float(v) for v in cfg.get("tool_offset_xyz", [0, 0, 262])]
                self.soft_limit_enabled = bool(cfg.get("soft_limit_enabled", DEFAULT_SOFT_LIMIT_ENABLED))
                self.z_min_limit = float(cfg.get("z_min_limit", DEFAULT_Z_MIN_LIMIT))
            except Exception as e:
                self.log(f"⚠️ 读取运行参数失败，改用默认值: {e}")
                self.manual_comp = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                self.tool_comp_enabled = True
                self.tool_offset_xyz = [0.0, 0.0, 262.0]
                self.soft_limit_enabled = DEFAULT_SOFT_LIMIT_ENABLED
                self.z_min_limit = DEFAULT_Z_MIN_LIMIT

        if hasattr(self, "manual_inputs"):
            for label, value in zip(["X", "Y", "Z", "Rx", "Ry", "Rz"], self.manual_comp):
                self.manual_inputs[label].setText(f"{float(value):.3f}")
        if hasattr(self, "tool_inputs"):
            for label, value in zip(["X", "Y", "Z"], self.tool_offset_xyz):
                self.tool_inputs[label].setText(f"{float(value):.3f}")
        if hasattr(self, "input_z_min_limit"):
            self.input_z_min_limit.setText(f"{float(self.z_min_limit):.3f}")
        self.refresh_runtime_labels()

    def save_runtime_config(self):
        cfg = {
            "manual_comp": list(self.manual_comp),
            "tool_comp_enabled": bool(self.tool_comp_enabled),
            "tool_offset_xyz": list(self.tool_offset_xyz),
            "soft_limit_enabled": bool(self.soft_limit_enabled),
            "z_min_limit": float(self.z_min_limit),
        }
        with open(FILE_RUNTIME_CONFIG, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    def refresh_runtime_labels(self):
        if hasattr(self, "lbl_manual_status"):
            self.lbl_manual_status.setText(f"当前手动补偿：{comp_to_text(self.manual_comp)}")
        if hasattr(self, "lbl_tool_enable"):
            self.lbl_tool_enable.setText(
                f"当前工具补偿：{'启用' if self.tool_comp_enabled else '禁用'}\n"
                f"工具偏移 XYZ(mm)：({self.tool_offset_xyz[0]:.3f}, {self.tool_offset_xyz[1]:.3f}, {self.tool_offset_xyz[2]:.3f})"
            )
        if hasattr(self, "lbl_soft_limit_status"):
            self.lbl_soft_limit_status.setText(
                f"当前软限位：{'启用' if self.soft_limit_enabled else '禁用'} | "
                f"Z最小值={self.z_min_limit:.3f} mm\n"
                f"说明：该设置会持续生效，直到下一次修改。"
            )
        if hasattr(self, "lbl_runtime_status"):
            self.lbl_runtime_status.setText(
                f"本次运行手动补偿：{comp_to_text(self.manual_comp)}\n"
                f"TCP 工具补偿：{'启用' if self.tool_comp_enabled else '禁用'} | "
                f"XYZ=({self.tool_offset_xyz[0]:.3f}, {self.tool_offset_xyz[1]:.3f}, {self.tool_offset_xyz[2]:.3f})\n"
                f"Z轴软限位：{'启用' if self.soft_limit_enabled else '禁用'} | Z最小值={self.z_min_limit:.3f} mm\n"
                f"ROI：仅本次运行有效，重启后恢复默认范围\n"
                f"说明：补偿、软限位会持续生效，ROI不保存上一次记录。"
            )

    def apply_runtime_settings_to_video_thread(self):
        try:
            if self.vt:
                self.vt.set_manual_compensation(self.manual_comp)
                self.vt.set_tool_compensation(self.tool_comp_enabled, self.tool_offset_xyz)
                if hasattr(self.vt, "reload_hand_eye_matrix"):
                    self.vt.reload_hand_eye_matrix()
            if self.ct:
                self.ct.set_soft_limit(self.soft_limit_enabled, self.z_min_limit)
        except Exception as e:
            self.log(f"⚠️ 同步运行补偿到视觉线程失败: {e}")
        self.refresh_runtime_labels()

    def read_manual_comp_from_ui(self):
        vals = []
        for label in ["X", "Y", "Z", "Rx", "Ry", "Rz"]:
            text = self.manual_inputs[label].text().strip()
            vals.append(float(text) if text else 0.0)
        return tuple(vals)

    def apply_manual_comp_from_ui(self):
        try:
            self.manual_comp = self.read_manual_comp_from_ui()
            self.save_runtime_config()
            self.apply_runtime_settings_to_video_thread()
            self.log(f"✅ 已应用手动补偿: {comp_to_text(self.manual_comp)}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"应用手动补偿失败: {e}")

    def clear_manual_comp(self):
        self.manual_comp = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for label, value in zip(["X", "Y", "Z", "Rx", "Ry", "Rz"], self.manual_comp):
            self.manual_inputs[label].setText(f"{value:.3f}")
        self.save_runtime_config()
        self.apply_runtime_settings_to_video_thread()
        self.log("↺ 已清零手动补偿")

    def read_tool_comp_from_ui(self):
        vals = []
        for label in ["X", "Y", "Z"]:
            text = self.tool_inputs[label].text().strip()
            vals.append(float(text) if text else 0.0)
        return vals

    def apply_tool_comp_from_ui(self):
        try:
            self.tool_offset_xyz = self.read_tool_comp_from_ui()
            self.save_runtime_config()
            self.apply_runtime_settings_to_video_thread()
            self.log(
                f"✅ 已应用工具补偿偏移: "
                f"({self.tool_offset_xyz[0]:.3f}, {self.tool_offset_xyz[1]:.3f}, {self.tool_offset_xyz[2]:.3f})"
            )
        except Exception as e:
            QMessageBox.critical(self, "错误", f"应用工具偏移失败: {e}")

    def toggle_tool_compensation(self):
        self.tool_comp_enabled = not self.tool_comp_enabled
        self.save_runtime_config()
        self.apply_runtime_settings_to_video_thread()
        self.log(f"🔁 工具补偿已切换为：{'启用' if self.tool_comp_enabled else '禁用'}")

    def apply_soft_limit_from_ui(self):
        try:
            text = self.input_z_min_limit.text().strip()
            if not text:
                raise ValueError("Z最小值不能为空")
            self.z_min_limit = float(text)
            self.save_runtime_config()
            self.apply_runtime_settings_to_video_thread()
            self.log(f"✅ 已应用Z轴软限位数值: {self.z_min_limit:.3f} mm")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"应用软限位失败: {e}")

    def toggle_soft_limit(self):
        target = not self.soft_limit_enabled
        reply = QMessageBox.warning(
            self,
            "软限位开关警告",
            f"⚠️ 即将{'启用' if target else '禁用'}Z轴软限位。\n\n"
            "软限位会影响发送给机械臂的Z高度，错误设置可能导致碰撞或取放异常。\n"
            "请确认只有维护人员在安全状态下操作。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        self.soft_limit_enabled = target
        self.save_runtime_config()
        self.apply_runtime_settings_to_video_thread()
        self.log(f"🔁 Z轴软限位已切换为：{'启用' if self.soft_limit_enabled else '禁用'}")

    def enter_calib_mode(self):
        if os.path.exists(FILE_CAM): shutil.copy(FILE_CAM, FILE_CAM_TEMP)
        if os.path.exists(FILE_ROBOT): shutil.copy(FILE_ROBOT, FILE_ROBOT_TEMP)

        reply = QMessageBox.question(self, "开始标定",
                                     "【重要】您想要开始新的标定流程（清空旧数据），还是继续上次的标定？\n\nYes = 清空旧数据，重新开始\nNo = 保留旧数据，继续添加\nCancel = 取消，返回主界面",
                                     QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)

        if reply == QMessageBox.Yes:
            open(FILE_CAM, 'w').close()
            open(FILE_ROBOT, 'w').close()
            self.log("🗑️ 旧标定数据已清空，开始新会话")
            self.lbl_cam_status.setText("等待采集...")
            self.lbl_rob_status.setText("等待输入...")
            self.calib_is_dirty = True
            self.calib_calculated = False
            self.stack.setCurrentIndex(1)
            self.vt.mode = "CALIB"

        elif reply == QMessageBox.No:
            self.log("📂 保留旧数据，继续标定")
            c_cam = get_next_index(FILE_CAM) - 1
            c_rob = get_next_index(FILE_ROBOT) - 1
            self.lbl_cam_status.setText(f"📁 现有 {c_cam} 个视觉点")
            self.lbl_rob_status.setText(f"📁 现有 {c_rob} 个机械臂点")
            self.calib_is_dirty = False
            self.calib_calculated = False
            self.stack.setCurrentIndex(1)
            self.vt.mode = "CALIB"
        else:
            self.log("用户取消了进入标定模式")
            return

    def exit_calib_mode(self):
        if self.calib_is_dirty and not self.calib_calculated:
            reply = QMessageBox.warning(self, "警告",
                                        "检测到数据已修改但【未点击计算】。\n\n现在的返回操作将丢弃所有修改，恢复到进入前的状态。\n是否确认返回？",
                                        QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                if os.path.exists(FILE_CAM_TEMP): shutil.copy(FILE_CAM_TEMP, FILE_CAM)
                if os.path.exists(FILE_ROBOT_TEMP): shutil.copy(FILE_ROBOT_TEMP, FILE_ROBOT)
                self.log("↩️ 未计算，已回滚到进入前的状态")
            else:
                return

        self.stack.setCurrentIndex(0)
        self.vt.mode = "DETECT"

    def overwrite_original_backup(self):
        reply = QMessageBox.question(
            self,
            "覆盖原始备份",
            "确定要用当前最新标定数据覆盖之前的兜底原始备份吗？\n\n"
            "覆盖后，后续点击“恢复原始备份(兜底)”将恢复到这一次的数据。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        try:
            if not os.path.exists(FILE_CAM) or not os.path.exists(FILE_ROBOT):
                QMessageBox.warning(self, "提示", "当前标定数据文件不完整，无法覆盖原始备份。")
                return

            shutil.copy(FILE_CAM, FILE_CAM_ORIGIN)
            shutil.copy(FILE_ROBOT, FILE_ROBOT_ORIGIN)
            self.log("💾 已用当前最新标定数据覆盖原始兜底备份")
            QMessageBox.information(self, "成功", "已覆盖原始备份。以后恢复兜底时会使用当前这份标定数据。")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"覆盖原始备份失败: {e}")

    def restore_original_backup(self):
        reply = QMessageBox.question(self, "恢复原始备份",
                                     "⚠️ 确定要丢弃当前所有标定数据，并恢复到【原始/出厂】备份吗？\n此操作不可撤销，并会立即尝试重新计算。",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            if os.path.exists(FILE_CAM_ORIGIN) and os.path.exists(FILE_ROBOT_ORIGIN):
                shutil.copy(FILE_CAM_ORIGIN, FILE_CAM)
                shutil.copy(FILE_ROBOT_ORIGIN, FILE_ROBOT)
                self.log("♻️ 已恢复原始备份数据")
                self.lbl_cam_status.setText("♻️ 已恢复原始备份")
                self.lbl_rob_status.setText("♻️ 已恢复原始备份")
                self.calc()
            else:
                QMessageBox.critical(self, "错误", "未找到原始备份文件！")

    def capture_camera_point(self):
        poses = getattr(self.vt, "cur_poses", None)
        if poses and len(poses) > 0:
            p = poses[0]
            idx = get_next_index(FILE_CAM)
            # 只保存 6 列：X,Y,Z,Rx,Ry,Rz；不再写入序号，避免计算脚本按 6 列读取时跳过整行
            line = f"{p['position'][0] * 1000:.3f},{p['position'][1] * 1000:.3f},{p['position'][2] * 1000:.3f},{p['euler_angles'][0]:.3f},{p['euler_angles'][1]:.3f},{p['euler_angles'][2]:.3f}\n"
            with open(FILE_CAM, "a", encoding='utf-8') as f:
                f.write(line)
            self.lbl_cam_status.setText(f"✅ 第 {idx} 个视觉点已保存")
            self.log(f"视觉点采集成功: {line.strip()}")
            QMessageBox.information(self, "提示", f"第 {idx} 个点采集成功！\n请移动机械臂对准中心，并在右侧输入坐标。")
            self.calib_is_dirty = True
        else:
            self.lbl_cam_status.setText("❌ 未检测到标定板")
            QMessageBox.warning(self, "警告", "画面中未检测到 AprilTag 标定板！")

    def save_robot_point(self):
        try:
            vals = []
            for label in ['X', 'Y', 'Z', 'Rx', 'Ry', 'Rz']:
                text = self.inputs[label].text().strip()
                if not text: raise ValueError(f"{label} 不能为空")
                vals.append(text)

            idx = get_next_index(FILE_ROBOT)
            # 只保存 6 列：X,Y,Z,Rx,Ry,Rz；不再写入序号
            line = ",".join(vals) + "\n"

            with open(FILE_ROBOT, "a", encoding='utf-8') as f:
                f.write(line)
            self.log(f"机械臂点保存成功: {line.strip()}")

            self.lbl_rob_status.setText(f"✅ 第 {idx} 个机械臂坐标已保存")

            QMessageBox.information(self, "成功", f"第 {idx} 个机械臂坐标已保存！")
            for inp in self.inputs.values(): inp.clear()
            self.calib_is_dirty = True
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {e}")

    def open_file_editor(self, title, filename):
        dialog = FileEditorDialog(title, filename, self)
        dialog.exec()

    def view_calib_result(self):
        if os.path.exists(FILE_T_BASE_CAM_TXT):
            self.open_file_editor("标定计算结果 (T_base_cam.txt)", FILE_T_BASE_CAM_TXT)
        else:
            QMessageBox.information(self, "提示", "尚未生成标定结果，请先完成采集并点击底部的计算按钮。")

    def log(self, m):
        # 限制GUI日志行数，防止长时间运行时 QTextEdit 内容无限增长导致内存上涨
        try:
            self.txt_log.document().setMaximumBlockCount(MAX_GUI_LOG_LINES)
        except Exception:
            pass

        self.txt_log.append(f"{time.strftime('%H:%M:%S')} {m}")
        self.txt_log.verticalScrollBar().setValue(
            self.txt_log.verticalScrollBar().maximum())

    def update_img(self, img):
        if self.stack.currentIndex() == 2: return
        h, w, c = img.shape
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb_img.data, w, h, c * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        if self.stack.currentIndex() == 0:
            if hasattr(self.lbl_v1, "set_video_pixmap"):
                self.lbl_v1.set_video_pixmap(pix)
            else:
                self.lbl_v1.setPixmap(pix)
        else:
            if hasattr(self.lbl_v2, "set_video_pixmap"):
                self.lbl_v2.set_video_pixmap(pix)
            else:
                self.lbl_v2.setPixmap(pix)

    def switch_mode(self, m):
        self.stack.setCurrentIndex(0)
        self.vt.mode = "DETECT"

    def start_roi_edit(self):
        if not self.vt.rs_cam: return
        color, _, _ = self.vt.rs_cam.capture_frames()
        if color is None: QMessageBox.warning(self, "错误", "无法获取图像"); return
        self.page_roi.set_image(color)
        self.stack.setCurrentIndex(2)
        self.vt.mode = "PAUSE"

    def on_roi_updated(self, roi):
        self.vt.rs_cam.set_roi(roi)
        self.log(f"✅ ROI更新: {roi}")
        self.switch_mode("DETECT")

    def on_roi_cancelled(self):
        self.log("取消ROI修改")
        self.switch_mode("DETECT")

    def on_robot_trigger(self, cid):
        self.log("📸 收到拍照指令，执行单帧最高点识别...")
        pose, msg = self.vt.trigger_detection()
        if pose:
            self.log(f"✅ {msg}，准备发送坐标序列")
            self.ct.send_success_sequence(cid, pose)
        else:
            self.log(f"⚠️ {msg}，发送反馈NG")
            self.ct.send_feedback(cid, False)

    def do_plane(self):
        reply = QMessageBox.question(self, "修改平面", "需更改平面？请清空桌面后确认。",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            if self.vt.rs_cam:
                _, d, _ = self.vt.rs_cam.capture_frames()
                if self.vt.rs_cam.calibrate_table_normal(d):
                    self.log("✅ 平面更新")
                    QMessageBox.information(self, "成功", "平面已更新！")
                else:
                    self.log("❌ 失败")
                    QMessageBox.warning(self, "失败", "标定失败")

    def cap_pt(self):
        pass

    def calc(self):
        old_text = self.btn_calc_final.text()
        self.btn_calc_final.setEnabled(False)
        self.btn_calc_final.setText("正在计算中...")
        QApplication.processEvents()
        try:
            from hand_eye_calibration import main as calc_main
            ok = calc_main()
            if ok is not True:
                raise RuntimeError("计算脚本未返回成功状态，请检查控制台输出。")
            QMessageBox.information(self, "成功", f"计算完成！\n{FILE_T_BASE_CAM_NPY} 已更新。")
            if self.vt.rs_cam:
                self.vt.rs_cam.load_calibration_matrix()
            self.apply_runtime_settings_to_video_thread()
            self.calib_calculated = True
            self.calib_is_dirty = False
        except Exception as e:
            QMessageBox.critical(self, "计算失败", str(e))
        finally:
            self.btn_calc_final.setText(old_text)
            self.btn_calc_final.setEnabled(True)

    def closeEvent(self, e):
        try:
            self.save_runtime_config()
        except Exception:
            pass
        self.vt.stop()
        self.ct.stop()
        e.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())