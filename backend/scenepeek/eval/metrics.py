"""Retrieval metrics over timestamp ranges."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Range:
    video: str
    start_s: float
    end_s: float


def overlaps(hit: Range, rel: Range, tolerance_s: float = 3.0) -> bool:
    if hit.video != rel.video:
        return False
    if hit.start_s < rel.end_s and hit.end_s > rel.start_s:
        return True
    return rel.start_s - tolerance_s <= hit.start_s <= rel.end_s + tolerance_s


def relevance(hits: list[Range], relevant: list[Range], tolerance_s: float = 3.0) -> list[int]:
    """1 if the hit at that rank matches any relevant range. Each relevant range is credited once so
    duplicate hits on the same moment don't inflate recall."""
    remaining = list(relevant)
    out = []
    for h in hits:
        matched = next((r for r in remaining if overlaps(h, r, tolerance_s)), None)
        if matched is not None:
            remaining.remove(matched)
            out.append(1)
        else:
            out.append(0)
    return out


def recall_at(rels: list[int], n_relevant: int, k: int) -> float:
    if n_relevant == 0:
        return 0.0
    return min(1.0, sum(rels[:k]) / n_relevant)


def hit_at(rels: list[int], k: int) -> float:
    return 1.0 if any(rels[:k]) else 0.0


def mrr(rels: list[int]) -> float:
    for i, r in enumerate(rels, start=1):
        if r:
            return 1.0 / i
    return 0.0


def ndcg_at(rels: list[int], n_relevant: int, k: int) -> float:
    dcg = sum(r / math.log2(i + 1) for i, r in enumerate(rels[:k], start=1))
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(n_relevant, k) + 1))
    return dcg / ideal if ideal else 0.0


def lane_attribution(
    lanes: dict[str, list[Range]], relevant: list[Range], tolerance_s: float = 3.0, k: int = 10
) -> dict:
    """Which candidate lanes surfaced a relevant span, and how deep. Explains *why* fusion helps:
    a query found only by the OCR lane is evidence the lane earns its keep."""
    per_lane: dict[str, dict] = {}
    for lane, hits in lanes.items():
        rels = relevance(hits[:k], relevant, tolerance_s)
        first = next((i for i, r in enumerate(rels, start=1) if r), None)
        per_lane[lane] = {"first_rank": first, f"hit@{k}": 1.0 if first else 0.0}
    found = [lane for lane, v in per_lane.items() if v["first_rank"]]
    return {
        "lanes": per_lane,
        "ceiling": bool(found),  # some lane had it in its top-k, so fusion *could* have surfaced it
        "unique": found[0] if len(found) == 1 else None,
    }


# -- temporal grounding (IoU-based, as used by QVHighlights / Charades-STA / ActivityNet) ---------


def iou(hit: Range, rel: Range) -> float:
    if hit.video != rel.video:
        return 0.0
    inter = max(0.0, min(hit.end_s, rel.end_s) - max(hit.start_s, rel.start_s))
    union = (hit.end_s - hit.start_s) + (rel.end_s - rel.start_s) - inter
    return inter / union if union > 0 else 0.0


def recall_at_iou(hits: list[Range], relevant: list[Range], k: int, thr: float) -> float:
    """R@k,IoU>=thr: 1 if any of the top-k predictions overlaps some ground-truth window at >= thr."""
    return 1.0 if any(iou(h, r) >= thr for h in hits[:k] for r in relevant) else 0.0


def average_precision_at_iou(hits: list[Range], relevant: list[Range], thr: float) -> float:
    """AP over a ranked list with multiple ground-truth windows: a prediction is a true positive if it
    matches a not-yet-matched GT window at IoU >= thr (greedy, highest-IoU window first)."""
    if not relevant:
        return 0.0
    remaining = list(relevant)
    tp, precisions = 0, []
    for i, h in enumerate(hits, start=1):
        best = max(remaining, key=lambda r: iou(h, r), default=None)
        if best is not None and iou(h, best) >= thr:
            remaining.remove(best)
            tp += 1
            precisions.append(tp / i)
    return sum(precisions) / len(relevant)


def mean_ap(hits: list[Range], relevant: list[Range], thresholds: tuple[float, ...] | None = None) -> float:
    """QVHighlights-style mAP averaged over IoU thresholds 0.5:0.05:0.95."""
    thresholds = thresholds or tuple(round(0.5 + 0.05 * i, 2) for i in range(10))
    return sum(average_precision_at_iou(hits, relevant, t) for t in thresholds) / len(thresholds)


def temporal_metrics(hits: list[Range], relevant: list[Range]) -> dict[str, float]:
    return {
        "r1@0.5": recall_at_iou(hits, relevant, 1, 0.5),
        "r1@0.7": recall_at_iou(hits, relevant, 1, 0.7),
        "r5@0.5": recall_at_iou(hits, relevant, 5, 0.5),
        "r5@0.7": recall_at_iou(hits, relevant, 5, 0.7),
        "map@0.5": average_precision_at_iou(hits, relevant, 0.5),
        "map@0.75": average_precision_at_iou(hits, relevant, 0.75),
        "map": mean_ap(hits, relevant),
    }
