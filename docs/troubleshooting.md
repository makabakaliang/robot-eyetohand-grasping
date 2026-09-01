# Troubleshooting

## `pyrealsense2` import fails
- Install the Intel RealSense SDK first, then `pip install pyrealsense2`.
- On Jetson, use the JetPack-matched wheel rather than PyPI.

## YOLO weights not found
- The repo does **not** ship `.pt` weights (see `.gitignore`). Train with
  `training/yolo-segmentation/train.py` or place your own weights under
  `weights/` and point `detect.py` at them.

## Calibration RMS too large (>5 mm)
- Re-capture more pose pairs, covering a larger volume.
- Make sure the AprilTag board is fully visible and not motion-blurred.
- Check that robot base poses are read from the same coordinate convention.

## GUI cannot connect to robot
- Start `mock_robot.py` first if you are developing without hardware.
- Verify firewall allows TCP 9760, or change `HOST/PORT` in both ends.
- On the Jetson, the GUI is launched by `scripts/robot-gui.service`; check
  `journalctl --user -u robot-gui.service`.

## `wx` / display errors over SSH
- Set `export DISPLAY=:0` (handled by the systemd unit).
- For headless debugging, run `python detect.py` directly instead of the GUI.
