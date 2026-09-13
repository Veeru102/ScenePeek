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
