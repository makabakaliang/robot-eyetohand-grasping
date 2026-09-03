# Changelog

All notable changes to this project are documented here.

## [Unreleased]
### Added
- Unit tests for payload builders and calibration math.
- CI workflow running tests on Ubuntu / Python 3.10 and 3.11.
- `scripts/verify_install.py` for environment self-check.
- Architecture and troubleshooting docs.
- `runtime_config.example.json` template.
- MIT license.

## [0.1.0] - 2026-08-21
### Added
- Eye-to-hand grasping pipeline: RealSense + YOLOv8-seg + AprilTag calibration.
- PyQt5 control GUI with ROI, parameter editing and grasp trigger.
- TCP/JSON protocol helpers and mock robot server.
- YOLOv8-seg dataset pipeline (MakeSense → YOLO labels, split, augmentation).
