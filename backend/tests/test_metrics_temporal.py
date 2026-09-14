from scenepeek.eval.metrics import (
    Range,
    average_precision_at_iou,
    iou,
    mean_ap,
    recall_at_iou,
    temporal_metrics,
)

V = "v"


def r(a, b, video=V):
    return Range(video, a, b)


def test_iou_basic_and_other_video():
    assert iou(r(0, 10), r(5, 15)) == 5 / 15
    assert iou(r(0, 10), r(0, 10)) == 1.0
    assert iou(r(0, 10), r(20, 30)) == 0.0
    assert iou(r(0, 10), r(0, 10, video="other")) == 0.0


def test_recall_at_iou_threshold_and_rank():
    hits = [r(50, 60), r(0, 10)]
    gt = [r(0, 12)]
    assert recall_at_iou(hits, gt, 1, 0.5) == 0.0  # rank-1 misses
    assert recall_at_iou(hits, gt, 5, 0.5) == 1.0  # rank-2 IoU = 10/12
    assert recall_at_iou(hits, gt, 5, 0.9) == 0.0


def test_average_precision_multi_window():
    gt = [r(0, 10), r(100, 110)]
    hits = [r(0, 10), r(40, 50), r(100, 110)]  # TP, FP, TP -> precisions 1/1, 2/3
    assert abs(average_precision_at_iou(hits, gt, 0.5) - (1.0 + 2 / 3) / 2) < 1e-9
    # a second hit on the same window is not a second true positive
    assert average_precision_at_iou([r(0, 10), r(1, 11)], gt, 0.5) == 0.5


def test_mean_ap_averages_thresholds():
    gt = [r(0, 10)]
    exact = mean_ap([r(0, 10)], gt)
    loose = mean_ap([r(0, 12)], gt)  # IoU 0.83: counts for thr <= 0.8 only
    assert exact == 1.0
    assert abs(loose - 0.7) < 1e-9


def test_temporal_metrics_keys():
    m = temporal_metrics([r(0, 10)], [r(0, 10)])
    assert set(m) == {"r1@0.5", "r1@0.7", "r5@0.5", "r5@0.7", "map@0.5", "map@0.75", "map"}
    assert all(v == 1.0 for v in m.values())
