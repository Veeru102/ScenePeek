"""ScenePeek's own YAML datasets (the hand-written human set, the auto-generated real set, the
synthetic regression set). Videos are local files already uploaded to the library; the importer
links them by title."""

import shutil
from collections.abc import Iterable
from pathlib import Path

import yaml

from scenepeek.datasets.base import QuerySpec, Unavailable, VideoSpec


class LocalYaml:
    version = "yaml"
    splits = ("test",)

    def __init__(self, name: str, file: str):
        self.name = name
        self.file = file

    def _data(self, root: Path) -> dict:
        return yaml.safe_load((root / self.file).read_text())

    def iter_videos(self, root: Path, split: str) -> Iterable[VideoSpec]:
        for v in self._data(root).get("videos", []):
            yield VideoSpec(
                external_id=v["key"],
                split=split,
                meta={"title": v["title"], "file": v.get("file"), "license": v.get("license")},
            )

    def iter_queries(self, root: Path, split: str) -> Iterable[QuerySpec]:
        for q in self._data(root).get("queries", []):
            rel = q.get("relevant", [])
            yield QuerySpec(
                external_id=str(q["id"]),
                video_external_id=rel[0]["video"],
                split=split,
                text=q["text"],
                relevant=[(float(r["start_s"]), float(r["end_s"])) for r in rel],
                modality=q.get("modality"),
                meta={"notes": q.get("notes"), "author": q.get("author")},
            )

    def fetch(self, video: VideoSpec, dest: Path) -> None:
        src = video.meta.get("file")
        if not src or not Path(src).exists():
            raise Unavailable(f"local file missing: {src}")
        shutil.copyfile(src, dest)
