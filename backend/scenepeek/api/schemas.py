import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class VideoCreate(BaseModel):
    filename: str
    content_type: str = "video/mp4"
    size_bytes: int | None = None
    title: str | None = None


class UploadTarget(BaseModel):
    video_id: uuid.UUID
    upload_url: str
    key: str


class ChunkOut(BaseModel):
    index: int
    start_s: float
    end_s: float
    status: str
    stage: str | None = None
    error: str | None = None


class VideoOut(BaseModel):
    id: uuid.UUID
    title: str
    status: str
    error: str | None
    duration_s: float | None
    width: int | None
    height: int | None
    size_bytes: int | None
    chunk_count: int
    chunks_done: int
    chunks_failed: int
    progress: float = Field(description="0..1 fraction of chunks indexed")
    poster_url: str | None
    playback_url: str | None
    created_at: datetime
    upload_completed_at: datetime | None
    first_searchable_at: datetime | None
    completed_at: datetime | None
    topic_count: int = 0


class VideoDetail(VideoOut):
    chunks: list[ChunkOut]


class UtteranceOut(BaseModel):
    start_s: float
    end_s: float
    text: str
    words: list | None = None


class FrameOut(BaseModel):
    t_s: float
    url: str
    caption: str = ""


class TopicOut(BaseModel):
    index: int
    start_s: float
    end_s: float
    label: str
    keyphrases: list[str] | None = None


class SearchWeights(BaseModel):
    text: float | None = None
    lexical: float | None = None
    visual: float | None = None
    ocr: float | None = None
    caption: float | None = None


class SearchRequest(BaseModel):
    q: str = Field(min_length=1, max_length=500)
    video_ids: list[uuid.UUID] | None = None
    limit: int = Field(default=20, ge=1, le=100)
    weights: SearchWeights | None = None
    rerank: bool | None = None
    fusion: str | None = None


class SearchVideo(BaseModel):
    id: uuid.UUID
    title: str
    duration_s: float | None
    poster_url: str | None
    playback_url: str | None


class SearchHit(BaseModel):
    segment_id: uuid.UUID
    video: SearchVideo
    start_s: float
    end_s: float
    score: float
    signals: dict[str, float]
    text: str
    snippet_html: str
    ocr_text: str
    caption_text: str = ""
    keyframe_url: str | None


class SearchResponse(BaseModel):
    query: str
    plan: dict
    hits: list[SearchHit]
    timings_ms: dict[str, float]
    total_candidates: int


class JobOut(BaseModel):
    id: uuid.UUID
    type: str
    queue: str
    status: str
    priority: int
    attempts: int
    max_attempts: int
    video_id: uuid.UUID | None
    payload: dict
    last_error: str | None
    locked_by: str | None
    run_after: datetime
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
