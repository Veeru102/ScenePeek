from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    database_url: str = "postgresql+asyncpg://scenepeek:scenepeek@localhost:5434/scenepeek"

    # Object storage
    s3_endpoint: str = "http://localhost:9000"
    s3_public_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "scenepeek"
    s3_secret_key: str = "scenepeek123"
    s3_bucket: str = "scenepeek"
    s3_region: str = "us-east-1"

    # API
    api_port: int = 8000
    cors_origins: str = "http://localhost:5173"

    # Worker
    worker_queues: str = "cpu,ml"
    worker_metrics_port: int = 9101
    worker_poll_interval_s: float = 1.0
    worker_heartbeat_s: float = 10.0
    worker_lease_timeout_s: float = 90.0
    media_cache_dir: Path = Path("~/.cache/scenepeek")

    # Chunking / segmentation
    chunk_seconds: float = 60.0
    segment_target_s: float = 10.0
    segment_min_s: float = 6.0
    segment_max_s: float = 15.0
    frame_sample_fps: float = 1.0
    frame_max_width: int = 480
    keyframes_per_segment: int = 3

    # Models
    whisper_model: str = "small.en"
    whisper_backend: str = "faster_whisper"  # faster_whisper | mlx_whisper (Apple, optional extra)
    whisper_device: str = "auto"  # auto | cpu | cuda  (CTranslate2 has no MPS backend)
    whisper_compute_type: str = "int8"  # CPU default; auto-promoted to float16 on CUDA
    text_embed_model: str = "BAAI/bge-small-en-v1.5"
    text_embed_dim: int = 384
    visual_embed_model: str = "google/siglip-base-patch16-224"
    visual_embed_dim: int = 768
    reranker_model: str = "BAAI/bge-reranker-base"
    ocr_enabled: bool = True
    ocr_min_conf: float = 0.5  # RapidOCR line confidence below which text is dropped at index time
    captions_enabled: bool = True
    caption_model: str = "Salesforce/blip-image-captioning-base"
    caption_max_tokens: int = 30
    caption_in_rerank_passage: bool = False
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    ollama_enabled: bool = False  # force LLM query decomposition / topic titles on
    ollama_auto: bool = True  # ...or use them automatically when a local Ollama server is running

    # Search
    router: str = "heuristic"  # fixed | heuristic | learned:<version> — how lane weights are chosen per query
    search_candidates: int = 200
    rerank_top_k: int = 30
    rerank_enabled: bool = True
    fusion_method: str = "weighted"  # weighted | rrf — weighted min-max measured +0.05 MRR on human queries
    rrf_k: int = 60
    weight_text: float = 1.0
    weight_lexical: float = 0.8
    weight_visual: float = 0.8
    weight_ocr: float = 0.6
    # captions are indexed (filmstrip / explain panel) but 0 here: BLIP-base captions measured
    # slightly negative for ranking on hand-written queries; raise once a stronger captioner lands
    weight_caption: float = 0.0
    # temporal (multi-frame) lane: off until the encoder has been backfilled and measured
    weight_temporal: float = 0.0
    temporal_model: str = "microsoft/xclip-base-patch32"
    temporal_window_s: float = 8.0
    temporal_stride_s: float = 4.0
    span_mode: str = "segment"  # segment | window — report a temporal-lane window as the hit span
    ocr_no_cue_factor: float = 1.0  # multiplier on weight_ocr when the query has no on-screen-text cue
    ocr_trgm_threshold: float = 0.45  # word_similarity floor for the OCR lane (noisy-OCR tolerance)
    dedup_window_s: float = 12.0
    max_hits_per_video: int = 3

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def queue_list(self) -> list[str]:
        return [q.strip() for q in self.worker_queues.split(",") if q.strip()]

    @property
    def sync_database_url(self) -> str:
        return self.database_url.replace("+asyncpg", "+psycopg")

    @property
    def cache_dir(self) -> Path:
        p = self.media_cache_dir.expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache
def get_settings() -> Settings:
    return Settings()
