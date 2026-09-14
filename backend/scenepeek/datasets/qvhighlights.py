"""QVHighlights (Lei et al., 2021): 150 s YouTube clips with free-form queries and 2 s-granular
moment windows. Annotations are the `highlight_{train,val}_release.jsonl` files from the
moment_detr release, placed under `--dir`; videos are fetched with yt-dlp on demand."""

import json
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

from scenepeek.datasets.base import QuerySpec, Unavailable, VideoSpec

FILES = {"train": "highlight_train_release.jsonl", "val": "highlight_val_release.jsonl"}
_UNAVAILABLE_MARKERS = (
    "Video unavailable",
    "Private video",
    "has been removed",
    "not available",
    "This video is no longer",
)


def parse_vid(vid: str) -> tuple[str, float, float]:
    """'NUsG9BgSes0_210.0_360.0' -> ('NUsG9BgSes0', 210.0, 360.0). YouTube ids may contain '_'."""
    yt, start, end = vid.rsplit("_", 2)
    return yt, float(start), float(end)


class QVHighlights:
    name = "qvhighlights"
    version = "release-2021"
    splits = tuple(FILES)

    def _rows(self, root: Path, split: str) -> Iterable[dict]:
        path = root / FILES[split]
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — download the moment_detr annotation release into {root}"
            )
        with path.open() as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    def iter_videos(self, root: Path, split: str) -> Iterable[VideoSpec]:
        seen: set[str] = set()
        for r in self._rows(root, split):
            if r["vid"] in seen:
                continue
            seen.add(r["vid"])
            yt, start, end = parse_vid(r["vid"])
            yield VideoSpec(
                external_id=r["vid"],
                split=split,
                duration_s=float(r.get("duration") or (end - start)),
                source_url=f"https://www.youtube.com/watch?v={yt}",
                meta={"youtube_id": yt, "clip_start_s": start, "clip_end_s": end},
            )

    def iter_queries(self, root: Path, split: str) -> Iterable[QuerySpec]:
        for r in self._rows(root, split):
            yield QuerySpec(
                external_id=str(r["qid"]),
                video_external_id=r["vid"],
                split=split,
                text=r["query"],
                relevant=[(float(a), float(b)) for a, b in r["relevant_windows"]],
                meta={"saliency_scores": r.get("saliency_scores"), "duration": r.get("duration")},
            )

    def fetch(self, video: VideoSpec, dest: Path) -> None:
        if shutil.which("yt-dlp") is None:
            raise RuntimeError("yt-dlp is not installed (uv sync --extra bench)")
        start, end = video.meta["clip_start_s"], video.meta["clip_end_s"]
        cmd = [
            "yt-dlp",
            "--quiet",
            "--no-warnings",
            "--no-playlist",
            "-f",
            "bv*[height<=480][ext=mp4]+ba[ext=m4a]/b[height<=480][ext=mp4]/b[height<=480]",
            "--download-sections",
            f"*{start:.0f}-{end:.0f}",
            "--force-keyframes-at-cuts",
            "--merge-output-format",
            "mp4",
            "-o",
            str(dest),
            video.source_url,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if proc.returncode != 0 or not dest.exists():
            err = (proc.stderr or proc.stdout)[-2000:]
            if any(m in err for m in _UNAVAILABLE_MARKERS):
                raise Unavailable(err)
            raise RuntimeError(f"yt-dlp failed: {err}")
