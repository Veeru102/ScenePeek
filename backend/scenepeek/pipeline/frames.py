"""Keyframe selection: sample at 1 fps, keep frames that differ visually from the last kept one."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image


@dataclass
class FrameRef:
    t_ms: int
    key: str
    phash: str

    @property
    def t_s(self) -> float:
        return self.t_ms / 1000.0


def phash(path: Path) -> str:
    import imagehash

    with Image.open(path) as im:
        return str(imagehash.phash(im))


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def select_keyframes(
    sampled: list[tuple[float, Path]], *, min_distance: int = 8, max_gap_s: float = 8.0
) -> list[tuple[float, Path, str]]:
    """Keep a frame when it differs from the last kept frame (scene change) or when `max_gap_s`
    has elapsed, so every ~10 s segment gets at least one representative frame."""
    kept: list[tuple[float, Path, str]] = []
    last_hash: str | None = None
    last_t = -1e9
    for t, p in sampled:
        h = phash(p)
        changed = last_hash is None or hamming(h, last_hash) >= min_distance
        if changed or (t - last_t) >= max_gap_s:
            kept.append((t, p, h))
            last_hash, last_t = h, t
    return kept


def pick_distinct(frames: list[FrameRef], k: int, min_distance: int = 6) -> list[FrameRef]:
    """Greedy: choose up to k frames that are mutually distinct by perceptual hash."""
    chosen: list[FrameRef] = []
    for f in frames:
        if all(hamming(f.phash, c.phash) >= min_distance for c in chosen):
            chosen.append(f)
        if len(chosen) >= k:
            break
    if not chosen and frames:
        chosen = [frames[len(frames) // 2]]
    return chosen


def dump_manifest(frames: list[FrameRef]) -> bytes:
    return json.dumps([asdict(f) for f in frames]).encode()


def load_manifest(data: bytes) -> list[FrameRef]:
    return [FrameRef(**d) for d in json.loads(data)]
