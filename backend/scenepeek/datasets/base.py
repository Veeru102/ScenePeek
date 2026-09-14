"""Benchmark adapters: turn a dataset's annotation files into ScenePeek's common representation
(videos with a split, queries with ground-truth windows) and know how to fetch each raw video, so
benchmarks run through the same probe -> extract -> index pipeline as user uploads."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class VideoSpec:
    external_id: str
    split: str
    duration_s: float | None = None
    source_url: str | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class QuerySpec:
    external_id: str
    video_external_id: str
    split: str
    text: str
    relevant: list[tuple[float, float]]
    modality: str | None = None
    meta: dict = field(default_factory=dict)


class Unavailable(Exception):
    """The raw video cannot be obtained (removed from YouTube, private, ...)."""


class BenchmarkAdapter(Protocol):
    name: str
    version: str
    splits: tuple[str, ...]

    def iter_videos(self, root: Path, split: str) -> Iterable[VideoSpec]: ...

    def iter_queries(self, root: Path, split: str) -> Iterable[QuerySpec]: ...

    def fetch(self, video: VideoSpec, dest: Path) -> None:
        """Download/copy the raw video to `dest` (an .mp4). Raise Unavailable when it cannot be."""
        ...
