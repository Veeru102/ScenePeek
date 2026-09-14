"""Eval dataset: videos (matched to the library by title) + queries with relevant time ranges."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from scenepeek.eval.metrics import Range


@dataclass
class DatasetVideo:
    key: str
    title: str
    file: str | None = None
    url: str | None = None
    license: str | None = None


@dataclass
class DatasetQuery:
    id: str
    text: str
    relevant: list[Range]
    modality: str = "speech"
    notes: str | None = None
    author: str | None = None  # "human" for hand-written queries, "auto" for generated ones


@dataclass
class Dataset:
    path: Path
    videos: list[DatasetVideo] = field(default_factory=list)
    queries: list[DatasetQuery] = field(default_factory=list)

    def video_by_key(self, key: str) -> DatasetVideo:
        return next(v for v in self.videos if v.key == key)


def load_dataset(path: str | Path) -> Dataset:
    p = Path(path)
    data = yaml.safe_load(p.read_text())
    ds = Dataset(path=p)
    for v in data.get("videos", []):
        ds.videos.append(DatasetVideo(**v))
    for q in data.get("queries", []):
        rel = [Range(r["video"], float(r["start_s"]), float(r["end_s"])) for r in q.get("relevant", [])]
        ds.queries.append(
            DatasetQuery(
                id=str(q["id"]),
                text=q["text"],
                relevant=rel,
                modality=q.get("modality", "speech"),
                notes=q.get("notes"),
                author=q.get("author"),
            )
        )
    return ds
