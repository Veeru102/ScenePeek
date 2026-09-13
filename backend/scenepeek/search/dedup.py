"""Temporal non-maximum suppression so one moment doesn't fill the results with neighbours."""

import uuid
from dataclasses import dataclass


@dataclass
class Hit:
    segment_id: uuid.UUID
    video_id: uuid.UUID
    start_s: float
    end_s: float
    score: float


def suppress(hits: list[Hit], window_s: float, max_per_video: int | None) -> list[Hit]:
    """`hits` sorted by score desc. A hit within `window_s` of a kept hit from the same video is
    dropped. Spans are never widened: the seek point must stay on the best-scoring segment."""
    kept: list[Hit] = []
    per_video: dict[uuid.UUID, int] = {}
    for h in hits:
        if any(
            k.video_id == h.video_id and h.start_s < k.end_s + window_s and h.end_s > k.start_s - window_s
            for k in kept
        ):
            continue
        if max_per_video and per_video.get(h.video_id, 0) >= max_per_video:
            continue
        per_video[h.video_id] = per_video.get(h.video_id, 0) + 1
        kept.append(h)
    return kept
