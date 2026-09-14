from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from scenepeek.api.schemas import SearchHit, SearchRequest, SearchResponse, SearchVideo
from scenepeek.core import storage
from scenepeek.core.db import get_session
from scenepeek.search.service import SearchOptions, highlight, search

router = APIRouter()


@router.post("", response_model=SearchResponse)
async def search_endpoint(body: SearchRequest, s: AsyncSession = Depends(get_session)):
    opts = SearchOptions(
        video_ids=body.video_ids,
        limit=body.limit,
        weights=body.weights.model_dump() if body.weights else None,
        rerank=body.rerank,
        fusion=body.fusion,
    )
    res = await search(s, body.q, opts)
    terms = res.plan.lexical_terms + res.plan.exact_phrases
    hits = []
    for h in res.hits:
        v = h.video
        hits.append(
            SearchHit(
                segment_id=h.segment.id,
                video=SearchVideo(
                    id=v.id,
                    title=v.title,
                    duration_s=v.duration_s,
                    poster_url=storage.presigned_get(v.poster_key) if v.poster_key else None,
                    playback_url=storage.presigned_get(v.web_key) if v.web_key else None,
                ),
                start_s=h.start_s,
                end_s=h.end_s,
                score=h.score,
                signals=h.signals,
                text=h.segment.text,
                snippet_html=highlight(h.segment.text, terms),
                ocr_text=h.segment.ocr_text,
                caption_text=h.segment.caption_text or "",
                keyframe_url=storage.presigned_get(h.segment.keyframe_key)
                if h.segment.keyframe_key
                else None,
            )
        )
    return SearchResponse(
        query=body.q,
        plan=res.plan.to_dict(),
        hits=hits,
        timings_ms=res.timings_ms,
        total_candidates=res.total_candidates,
    )
