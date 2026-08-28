"""Post-process detected candidates: depth gating, clustering, ranking."""
import numpy as np


def filter_by_depth(points, z_min=381.0, z_max=900.0):
    """Drop points whose camera-frame depth falls outside the working range.

    points: list of dicts with key 'xyz' -> (x, y, z).
    """
    return [p for p in points if z_min <= p["xyz"][2] <= z_max]


def nms_2d(candidates, iou_threshold=0.3):
    """Greedy 2D NMS on candidates that carry 'box' = (x1,y1,x2,y2)."""
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda c: c.get("score", 0.0), reverse=True)
    kept = []
    for c in ordered:
        if all(_iou(c["box"], k["box"]) < iou_threshold for k in kept):
            kept.append(c)
    return kept


def _iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / max(area_a + area_b - inter, 1e-6)


def rank_by_distance(points, target=(0.0, 0.0, 0.0)):
    """Sort candidates by Euclidean distance to the target point."""
    return sorted(points, key=lambda p: float(np.linalg.norm(np.array(p["xyz"]) - np.array(target))))
