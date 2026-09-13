"""Chunk window math. Chunks are the independent unit of processing."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkWindow:
    index: int
    start_s: float
    end_s: float


def plan_chunks(duration_s: float, chunk_s: float, min_tail_s: float = 15.0) -> list[ChunkWindow]:
    """Fixed windows; a short trailing remainder is merged into the last chunk."""
    if duration_s <= 0:
        return []
    n = max(1, int(duration_s // chunk_s))
    tail = duration_s - n * chunk_s
    if tail >= min_tail_s:
        n += 1
    out = []
    for i in range(n):
        start = i * chunk_s
        end = duration_s if i == n - 1 else (i + 1) * chunk_s
        out.append(ChunkWindow(index=i, start_s=round(start, 3), end_s=round(end, 3)))
    return out
