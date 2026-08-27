# Architecture

```
                ┌─────────────────────────────┐
                │        gui_app.py           │
                │  PyQt5: video, ROI, params,  │
                │  calibration capture, grasp  │
                └──────────┬──────────────────┘
                           │ commands / config
              ┌────────────┼────────────────────┐
              ▼            ▼                    ▼
        detect.py    json_payloads.py     runtime_config.json
   (RealSense+YOLO   (TCP/JSON protocol)  (tool offset, soft limits)
    depth→frame→base)
              │
              ▼
        T_base_cam.txt  ◄── hand_eye_calibration.py
        (fixed extrinsic)      (AprilTag poses → solve)
```

## Data flow per grasp

1. RealSense delivers RGB + aligned depth.
2. YOLOv8-seg returns HDD mask / center in image.
3. ROI depth average gives camera-frame (X_c, Y_c, Z_c).
4. Multiply by `T_base_cam` to get robot-base frame.
5. Apply TCP tool offset and manual compensation.
6. Send `AddRCC` / `AddPoints` JSON payload over TCP.

## Modules

| Module | Responsibility |
| --- | --- |
| `detect.py` | Vision pipeline: camera → detection → depth → frame transform |
| `hand_eye_calibration.py` | Solve `T_base_cam` from paired AprilTag / robot poses |
| `apritag_detect.py` | Capture calibration board poses during data collection |
| `gui_app.py` | Operator GUI; orchestrates the above without exposing internals |
| `json_payloads.py` | Encoder/decoder for the robot's TCP/JSON protocol |
| `mock_robot.py` | In-process fake robot for offline GUI development |
| `training/yolo-segmentation/` | Dataset conversion, augmentation, training entry |
