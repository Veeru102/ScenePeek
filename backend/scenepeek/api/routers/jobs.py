from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from scenepeek.api.schemas import JobOut
from scenepeek.core.db import get_session
from scenepeek.models import Job

router = APIRouter()


@router.get("", response_model=list[JobOut])
async def list_jobs(
    status: str | None = Query(None),
    limit: int = Query(50, le=500),
    s: AsyncSession = Depends(get_session),
):
    stmt = select(Job).order_by(Job.created_at.desc()).limit(limit)
    if status:
        stmt = stmt.where(Job.status == status)
    return list(await s.scalars(stmt))
