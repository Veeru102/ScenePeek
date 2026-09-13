import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from scenepeek import __version__
from scenepeek.api.routers import jobs, metrics, search, videos
from scenepeek.core.config import get_settings
from scenepeek.core.logging import configure_logging, get_logger
from scenepeek.core.storage import ensure_bucket

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    try:
        ensure_bucket()
    except Exception as e:  # storage may be starting up; don't block boot
        log.warning("bucket check failed", error=str(e))
    threading.Thread(target=_warm_models, daemon=True).start()
    yield


def _warm_models() -> None:
    """Load the query-side encoders in the background so the first search isn't a cold start."""
    try:
        from scenepeek.ml import reranker, siglip, text_embed

        text_embed.embed_query("warm up")
        siglip.embed_text("warm up")
        if get_settings().rerank_enabled:
            reranker.rerank_scores("warm up", ["warm up"])
        log.info("query encoders warm")
    except Exception as e:
        log.warning("model warmup failed", error=str(e))


app = FastAPI(title="ScenePeek API", version=__version__, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(videos.router, prefix="/api/videos", tags=["videos"])
app.include_router(search.router, prefix="/api/search", tags=["search"])
app.include_router(jobs.router, prefix="/api/jobs", tags=["jobs"])
app.include_router(metrics.router, prefix="/api/metrics", tags=["metrics"])
app.mount("/metrics", make_asgi_app())


@app.get("/health")
async def health():
    return {"status": "ok", "version": __version__}
