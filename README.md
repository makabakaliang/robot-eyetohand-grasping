# Bronte Robot Eye-to-Hand HDD Grasping

基于勃朗特机械臂、Intel RealSense 深度相机和 YOLO 的硬盘识别抓取项目。该版本采用 **eye-to-hand（相机固定在机械臂外部）** 方案，通过目标检测、深度定位、手眼标定和坐标变换，将硬盘在相机坐标系下的位置转换到机械臂基坐标系，并通过图形界面完成识别、标定、补偿和抓取流程。

## 项目亮点

- 实现从视觉识别到机械臂抓取的完整闭环流程。
- 使用 YOLO 对硬盘目标进行检测，并结合 RealSense 深度数据估计三维位置。
- 基于 AprilTag 标定板采集相机位姿与机械臂基座位姿，计算 `T_base_cam` 外参矩阵。
- 封装相机坐标系到机械臂基坐标系的转换、TCP 工具长度补偿和人工误差补偿。
- 使用 PyQt 构建控制界面，支持实时画面、ROI 设置、标定数据采集、参数编辑和抓取触发。
- 提供 mock robot 模块，便于在没有真实机械臂时调试通信流程。

## 技术栈

- Python
- OpenCV / OpenCV ArUco
- Intel RealSense SDK (`pyrealsense2`)
- Ultralytics YOLO
- NumPy / SciPy
- PyQt5
- TCP/JSON 机械臂通信

## 系统流程

1. RealSense 采集 RGB-D 图像。
2. YOLO 检测图像中的硬盘目标。
3. 在 ROI 内读取目标中心点深度，估算相机坐标系下的三维位姿。
4. 使用 AprilTag 标定数据求解相机到机械臂基座的外参 `T_base_cam`。
5. 将目标位姿转换到机械臂基坐标系。
6. 应用 TCP 工具补偿和人工微调补偿。
7. 通过 GUI 向机械臂发送查询、移动和抓取相关指令。

## 主要文件

| 文件 | 说明 |
| --- | --- |
| `gui_app.py` | PyQt 图形控制界面，集成视频显示、ROI、标定、参数编辑和机械臂通信 |
| `detect.py` | RealSense + YOLO 检测、深度定位、坐标转换和抓取目标计算 |
| `apritag_detect.py` | AprilTag / ArUco 标定板检测与相机位姿采集 |
| `hand_eye_calibration.py` | eye-to-hand 外参计算与误差评估 |
| `json_payloads.py` | 机械臂通信 JSON 指令构造 |
| `mock_robot.py` | 机械臂通信模拟服务 |
| `runtime_config.json` | 工具补偿、人工补偿和软限位等运行参数 |

## 训练与标定

- [YOLO 硬盘分割训练说明](docs/training.md)：包含 MakeSense 标注、YOLO 标签转换、数据划分、离线增强、训练配置和训练指标。
- [Eye-to-Hand 手眼标定说明](docs/calibration.md)：说明 AprilTag 位姿采集、`T_base_cam` 求解和坐标转换链路。

## Eye-to-Hand 方案特点

该方案中相机固定在工作空间外部，视野稳定，适合观察固定工位上的多个目标。标定结果主要描述相机坐标系到机械臂基坐标系的固定变换，因此抓取时不需要根据末端姿态实时更新相机外参。

相比 eye-in-hand 方案，它的优势是视野更稳定、目标搜索范围更大、控制流程更直接；不足是当机械臂或目标遮挡相机视线时，需要通过工位布局和 ROI 设置降低影响。

## 运行说明

安装依赖：

```bash
pip install -r requirements.txt
```

启动 GUI：

```bash
python gui_app.py
```

仅运行检测流程：

```bash
python detect.py
```

运行标定计算：

```bash
python hand_eye_calibration.py
```

## 上传说明

仓库中不建议提交本地虚拟环境、缓存、日志、模型权重和大型安装包。YOLO 权重、RealSense 设备、机械臂 IP、工具长度补偿等参数需要根据实际设备环境重新配置。
