"""Combine per-modality candidate lists into one score per segment."""

import uuid
from dataclasses import dataclass, field

from scenepeek.search.candidates import Cand


@dataclass
class Fused:
    segment_id: uuid.UUID
    fused: float
    signals: dict[str, float] = field(default_factory=dict)  # raw modality scores (for display)
    ranks: dict[str, int] = field(default_factory=dict)
    spans: dict[str, tuple[float, float]] = field(default_factory=dict)  # lane -> best window


def _minmax(cands: list[Cand]) -> dict[uuid.UUID, float]:
    if not cands:
        return {}
    lo, hi = min(c.score for c in cands), max(c.score for c in cands)
    if hi - lo < 1e-9:
        return {c.segment_id: 1.0 for c in cands}
    return {c.segment_id: (c.score - lo) / (hi - lo) for c in cands}


def fuse(
    lists: dict[str, list[Cand]], weights: dict[str, float], method: str = "rrf", rrf_k: int = 60
) -> list[Fused]:
    out: dict[uuid.UUID, Fused] = {}
    for mod, cands in lists.items():
        w = weights.get(mod, 0.0)
        normed = _minmax(cands)
        for rank, c in enumerate(cands, start=1):
            f = out.setdefault(c.segment_id, Fused(c.segment_id, 0.0))
            f.signals[mod] = c.score
            f.ranks[mod] = rank
            if c.span is not None:
                f.spans[mod] = c.span
            if method == "weighted":
                f.fused += w * normed[c.segment_id]
            else:
                f.fused += w / (rrf_k + rank)
    return sorted(out.values(), key=lambda f: f.fused, reverse=True)
