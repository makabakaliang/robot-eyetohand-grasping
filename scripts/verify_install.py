#!/usr/bin/env python3
"""Quick environment self-check: verify Python and key packages import.

Run this after setting up a new machine:
    python scripts/verify_install.py
"""
import importlib
import sys

REQUIRED = [
    ("numpy", "numpy"),
    ("cv2", "opencv-python"),
    ("scipy", "scipy"),
]

OPTIONAL = [
    ("pyrealsense2", "Intel RealSense SDK (only needed on robot workstation)"),
    ("ultralytics", "Ultralytics YOLO (only needed for inference/training)"),
    ("PyQt5", "PyQt5 (only needed for the GUI)"),
]


def main():
    print(f"Python {sys.version.split()[0]} on {sys.platform}")
    ok = True
    for mod, pkg in REQUIRED:
        try:
            m = importlib.import_module(mod)
            v = getattr(m, "__version__", "unknown")
            print(f"  [OK]  {mod:14s} {v}")
        except ImportError as e:
            print(f"  [FAIL] {mod:14s} not installed ({e}); pip install {pkg}")
            ok = False
    for mod, note in OPTIONAL:
        try:
            importlib.import_module(mod)
            print(f"  [OK]  {mod:14s} (optional)")
        except ImportError:
            print(f"  [ -- ] {mod:14s} not installed ({note})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
