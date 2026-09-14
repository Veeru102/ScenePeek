"""Per-modality candidate retrieval against Postgres (pgvector + FTS + trigram)."""

import uuid
from dataclasses import dataclass

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from scenepeek.core.config import get_settings


@dataclass
class Cand:
    segment_id: uuid.UUID
    score: float


def _vec(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v.tolist()) + "]"


def _filter(video_ids: list[uuid.UUID] | None, alias: str) -> str:
    return f" AND {alias}.video_id = ANY(:vids)" if video_ids else ""


async def by_text_vector(s: AsyncSession, q: np.ndarray, k: int, video_ids=None) -> list[Cand]:
    sql = text(
        f"""
        SELECT s.id, 1 - (s.text_embedding <=> CAST(:q AS vector)) AS score
        FROM segments s
        WHERE s.text_embedding IS NOT NULL {_filter(video_ids, "s")}
        ORDER BY s.text_embedding <=> CAST(:q AS vector)
        LIMIT :k
        """
    )
    rows = await s.execute(sql, {"q": _vec(q), "k": k, "vids": video_ids})
    return [Cand(r[0], float(r[1])) for r in rows]


async def by_caption(s: AsyncSession, q: np.ndarray, k: int, video_ids=None) -> list[Cand]:
    """Keyframe captions embedded with the text model: natural-language access to silent scenes."""
    sql = text(
        f"""
        SELECT s.id, 1 - (s.caption_embedding <=> CAST(:q AS vector)) AS score
        FROM segments s
        WHERE s.caption_embedding IS NOT NULL {_filter(video_ids, "s")}
        ORDER BY s.caption_embedding <=> CAST(:q AS vector)
        LIMIT :k
        """
    )
    rows = await s.execute(sql, {"q": _vec(q), "k": k, "vids": video_ids})
    return [Cand(r[0], float(r[1])) for r in rows]


async def by_visual_vector(s: AsyncSession, q: np.ndarray, k: int, video_ids=None) -> list[Cand]:
    """Nearest frames, max-pooled to their segment."""
    sql = text(
        f"""
        SELECT f.segment_id, MAX(1 - (f.visual_embedding <=> CAST(:q AS vector))) AS score
        FROM (
            SELECT segment_id, visual_embedding FROM frames
            WHERE visual_embedding IS NOT NULL {_filter(video_ids, "frames")}
            ORDER BY visual_embedding <=> CAST(:q AS vector)
            LIMIT :kf
        ) f
        GROUP BY f.segment_id
        ORDER BY score DESC
        LIMIT :k
        """
    )
    rows = await s.execute(sql, {"q": _vec(q), "k": k, "kf": k * 3, "vids": video_ids})
    return [Cand(r[0], float(r[1])) for r in rows]


def _websearch(terms: list[str], phrases: list[str]) -> str:
    parts = [f'"{p}"' for p in phrases] + terms
    return " OR ".join(parts)


async def by_lexical(
    s: AsyncSession, terms: list[str], phrases: list[str], k: int, video_ids=None
) -> list[Cand]:
    if not terms and not phrases:
        return []
    sql = text(
        f"""
        SELECT s.id, ts_rank_cd(s.text_tsv, q, 32) AS score
        FROM segments s, websearch_to_tsquery('english', :q) q
        WHERE s.text_tsv @@ q {_filter(video_ids, "s")}
        ORDER BY score DESC
        LIMIT :k
        """
    )
    rows = await s.execute(sql, {"q": _websearch(terms, phrases), "k": k, "vids": video_ids})
    return [Cand(r[0], float(r[1])) for r in rows]


async def by_ocr(
    s: AsyncSession, ocr_q: str, terms: list[str], phrases: list[str], k: int, video_ids=None
) -> list[Cand]:
    """FTS on the 'simple' config plus trigram word-similarity so noisy OCR still matches."""
    if not ocr_q.strip():
        return []
    sql = text(
        f"""
        SELECT s.id,
               GREATEST(ts_rank_cd(s.ocr_tsv, q, 32), word_similarity(:raw, s.ocr_text)) AS score
        FROM segments s, websearch_to_tsquery('simple', :q) q
        WHERE s.ocr_text <> '' AND (s.ocr_tsv @@ q OR word_similarity(:raw, s.ocr_text) > :trgm)
              {_filter(video_ids, "s")}
        ORDER BY score DESC
        LIMIT :k
        """
    )
    rows = await s.execute(
        sql,
        {
            "q": _websearch(terms, phrases) or ocr_q,
            "raw": ocr_q.lower(),
            "trgm": get_settings().ocr_trgm_threshold,
            "k": k,
            "vids": video_ids,
        },
    )
    return [Cand(r[0], float(r[1])) for r in rows]
