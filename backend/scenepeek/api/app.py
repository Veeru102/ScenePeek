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
    yield


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
