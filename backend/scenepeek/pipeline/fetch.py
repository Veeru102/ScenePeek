"""fetch_dataset_video (cpu queue): obtain a benchmark clip, register it as a library video, and
start the normal probe -> extract -> index pipeline at benchmark priority."""

from datetime import UTC, datetime

from scenepeek.core import storage
from scenepeek.datasets.base import Unavailable
from scenepeek.datasets.registry import get_adapter
from scenepeek.jobs import priority
from scenepeek.jobs import queue as q
from scenepeek.jobs.registry import JobContext
from scenepeek.models import Dataset, DatasetVideo, Video
from scenepeek.models.dataset import DatasetVideoStatus
from scenepeek.models.video import VideoStatus
from scenepeek.pipeline import media


def fetch_dataset_video(ctx: JobContext, payload: dict) -> None:
    adapter = get_adapter(payload["dataset"])
    with ctx.session() as s:
        dv = s.get(DatasetVideo, payload["dataset_video_id"])
        if dv is None:
            raise q.NonRetryableError("dataset video row missing")
        if dv.video_id is not None:
            return
        ds = s.get(Dataset, dv.dataset_id)
        spec = _spec(adapter, dv)
        dest = media.workdir("datasets", ds.name) / f"{dv.external_id}.mp4"

    try:
        if not dest.exists():
            tmp = dest.with_suffix(".part.mp4")
            adapter.fetch(spec, tmp)
            tmp.rename(dest)
    except Unavailable as e:
        with ctx.session() as s:
            dv = s.get(DatasetVideo, payload["dataset_video_id"])
            dv.status = DatasetVideoStatus.UNAVAILABLE
            dv.meta = {**dv.meta, "error": str(e)[-500:]}
            s.commit()
        raise q.NonRetryableError(f"unavailable: {str(e)[-200:]}") from e

    with ctx.session() as s:
        dv = s.get(DatasetVideo, payload["dataset_video_id"])
        video = Video(
            title=dv.external_id,
            original_key="",
            content_type="video/mp4",
            size_bytes=dest.stat().st_size,
            status=VideoStatus.UPLOADED,
            source=ds.name,
            priority_band=priority.BENCHMARK,
            upload_completed_at=datetime.now(UTC),
        )
        s.add(video)
        s.flush()
        video.original_key = storage.original_key(str(video.id), ".mp4")
        storage.upload_file(dest, video.original_key, "video/mp4")
        dv.video_id = video.id
        dv.status = DatasetVideoStatus.FETCHED
        q.enqueue_sync(
            s.connection(),
            "probe_video",
            {"video_id": str(video.id)},
            idempotency_key=f"probe:{video.id}",
            queue="cpu",
            priority=priority.BENCHMARK + 10,
            video_id=video.id,
        )
        s.commit()
    dest.unlink(missing_ok=True)
    ctx.log.info("dataset video fetched", dataset=ds.name, external_id=dv.external_id)


def _spec(adapter, dv: DatasetVideo):
    from scenepeek.datasets.base import VideoSpec

    return VideoSpec(
        external_id=dv.external_id,
        split=dv.split,
        duration_s=dv.duration_s,
        source_url=dv.source_url,
        meta=dict(dv.meta),
    )
