"""Tests for grab post-processing (NMS, depth filter, ranking)."""
from grab_postprocess import filter_by_depth, nms_2d, rank_by_distance


def test_depth_filter_drops_out_of_range():
    pts = [{"xyz": [0, 0, 400]}, {"xyz": [0, 0, 100]}, {"xyz": [0, 0, 800]}]
    kept = filter_by_depth(pts, z_min=381, z_max=900)
    assert len(kept) == 2


def test_nms_keeps_higher_score():
    cands = [
        {"box": [0, 0, 10, 10], "score": 0.9},
        {"box": [1, 1, 11, 11], "score": 0.5},
        {"box": [100, 100, 110, 110], "score": 0.8},
    ]
    kept = nms_2d(cands, iou_threshold=0.3)
    scores = sorted(c["score"] for c in kept)
    assert scores == [0.8, 0.9]


def test_rank_by_distance():
    pts = [{"xyz": [10, 0, 0]}, {"xyz": [1, 0, 0]}, {"xyz": [5, 0, 0]}]
    ranked = rank_by_distance(pts, target=(0, 0, 0))
    assert ranked[0]["xyz"] == [1, 0, 0]
