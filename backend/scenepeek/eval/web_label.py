"""Hand-labeling UI: watch a real video, mark a span, write the query *you* would type.

Writes only eval/real/human.yaml. Never touches candidates.yaml / dataset.yaml, and never
writes on GET, so a forgotten server can't clobber anything.
"""

from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from scenepeek.eval.real import REPO_ROOT, SOURCES_PATH, load_sources

HUMAN_PATH = REPO_ROOT / "eval/real/human.yaml"
_PAGE = Path(__file__).with_suffix(".html").read_text()
MODALITIES = ("speech", "visual", "ocr", "multi")
TARGETS = {"speech": 10, "visual": 8, "ocr": 6, "multi": 6}

app = FastAPI()


class LabelIn(BaseModel):
    video: str
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)
    text: str = Field(min_length=3, max_length=500)
    modality: str
    notes: str | None = None


def _load() -> dict:
    if not HUMAN_PATH.exists():
        return {"videos": [], "queries": []}
    return yaml.safe_load(HUMAN_PATH.read_text()) or {"videos": [], "queries": []}


def _save(data: dict) -> None:
    HUMAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    HUMAN_PATH.write_text(yaml.dump(data, sort_keys=False, allow_unicode=True, width=100))


def _counts(data: dict) -> dict[str, int]:
    out = dict.fromkeys(MODALITIES, 0)
    for q in data["queries"]:
        out[q.get("modality", "speech")] = out.get(q.get("modality", "speech"), 0) + 1
    return out


@app.get("/api/sources")
def sources():
    data = _load()
    return {
        "videos": [
            {"key": v["key"], "title": v["title"], "category": v.get("category")}
            for v in load_sources(SOURCES_PATH)
        ],
        "counts": _counts(data),
        "targets": TARGETS,
        "queries": data["queries"],
    }


@app.get("/api/video/{video_key}")
def video(video_key: str):
    info = next((v for v in load_sources(SOURCES_PATH) if v["key"] == video_key), None)
    if info is None:
        raise HTTPException(404, "unknown video key")
    path = REPO_ROOT / info["file"]
    if not path.exists():
        raise HTTPException(404, "video file missing")
    return FileResponse(path, media_type="video/mp4")


@app.post("/api/human")
def add_label(body: LabelIn):
    if body.modality not in MODALITIES:
        raise HTTPException(400, f"modality must be one of {MODALITIES}")
    if body.end_s <= body.start_s:
        raise HTTPException(400, "end_s must be after start_s")
    src = {v["key"]: v for v in load_sources(SOURCES_PATH)}
    if body.video not in src:
        raise HTTPException(404, "unknown video key")

    data = _load()
    if body.video not in {v["key"] for v in data["videos"]}:
        sv = src[body.video]
        data["videos"].append(
            {"key": sv["key"], "title": sv["title"], "file": sv["file"], "license": sv.get("license")}
        )
    n = sum(1 for q in data["queries"] if q["id"].startswith(f"human.{body.video}.")) + 1
    q = {
        "id": f"human.{body.video}.{n}",
        "text": body.text.strip(),
        "modality": body.modality,
        "author": "human",
        "relevant": [{"video": body.video, "start_s": round(body.start_s, 2), "end_s": round(body.end_s, 2)}],
    }
    if body.notes:
        q["notes"] = body.notes.strip()
    data["queries"].append(q)
    _save(data)
    return {"saved": q, "counts": _counts(data)}


@app.delete("/api/human/{query_id}")
def delete_label(query_id: str):
    data = _load()
    before = len(data["queries"])
    data["queries"] = [q for q in data["queries"] if q["id"] != query_id]
    if len(data["queries"]) == before:
        raise HTTPException(404, "no such query")
    _save(data)
    return {"counts": _counts(data)}


@app.get("/", response_class=HTMLResponse)
def index():
    return _PAGE
