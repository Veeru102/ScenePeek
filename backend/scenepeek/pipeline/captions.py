"""Backfill keyframe captions for already-indexed videos (no reindex, search stays live)."""

import time
from datetime import UTC, datetime

from PIL import Image
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from scenepeek.core import storage
from scenepeek.core.config import get_settings
from scenepeek.core.logging import get_logger
from scenepeek.ml import captioner, text_embed
from scenepeek.models import Frame, Segment, Video, VideoChunk
from scenepeek.pipeline import media

log = get_logger("captions")


def backfill(video_id: str | None = None, force: bool = False) -> dict[str, int]:
    settings = get_settings()
    engine = create_engine(settings.sync_database_url)
    done = {"segments": 0, "frames": 0, "videos": 0}
    with Session(engine) as s:
        vids = list(s.scalars(select(Video.id).where(Video.id == video_id) if video_id else select(Video.id)))
    for vid in vids:
        n = _backfill_video(engine, str(vid), force, settings)
        if n:
            done["videos"] += 1
            done["segments"] += n
    log.info("caption backfill complete", **done)
    return done


def _backfill_video(engine, video_id: str, force: bool, settings) -> int:
    t0 = time.perf_counter()
    total = 0
    with Session(engine) as s:
        chunks = list(
            s.scalars(select(VideoChunk).where(VideoChunk.video_id == video_id).order_by(VideoChunk.index))
        )
        title = s.scalar(select(Video.title).where(Video.id == video_id))
    for chunk in chunks:
        with Session(engine) as s:
            q = select(Segment).where(Segment.chunk_id == chunk.id).order_by(Segment.index)
            if not force:
                q = q.where(Segment.caption_text == "")
            segs = list(s.scalars(q))
            if not segs:
                continue
            frames = list(s.scalars(select(Frame).where(Frame.segment_id.in_([x.id for x in segs]))))
            by_seg: dict = {}
            for f in frames:
                by_seg.setdefault(f.segment_id, []).append(f)

            # caption each distinct image once
            uniq: dict[str, Frame] = {}
            for f in frames:
                uniq.setdefault(f.phash or f.image_key, f)
            images, keys = [], []
            for key, f in uniq.items():
                path = media.workdir(video_id, "captions") / f"{f.id}.jpg"
                if not path.exists():
                    storage.download_file(f.image_key, path)
                images.append(Image.open(path).convert("RGB"))
                keys.append(key)
            caps = dict(zip(keys, captioner.caption_images(images), strict=True))
            for im in images:
                im.close()

            texts = []
            for seg in segs:
                seg_frames = sorted(by_seg.get(seg.id, []), key=lambda f: f.t_s)
                for f in seg_frames:
                    f.caption = caps.get(f.phash or f.image_key, "")
                texts.append(captioner.merge_captions([f.caption for f in seg_frames]))
            vecs = text_embed.embed_passages([t if t else " " for t in texts]) if any(texts) else None
            for seg, t, i in zip(segs, texts, range(len(segs)), strict=True):
                seg.caption_text = t
                seg.caption_embedding = vecs[i].tolist() if (vecs is not None and t) else None
            chunk_row = s.get(VideoChunk, chunk.id)
            chunk_row.captioned_at = datetime.now(UTC)
            s.commit()
            total += len(segs)
    if total:
        log.info("video captioned", title=title, segments=total, seconds=round(time.perf_counter() - t0, 1))
    return total
