import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user, require_admin
from app.core.deps import get_db
from app.models.database import User, Video
from app.models.schemas import (
    IngestionStatusResponse,
    IngestionTriggerResponse,
    VideoListResponse,
    VideoResponse,
)
from app.services.ingestion import IngestionService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("", response_model=VideoListResponse)
async def list_videos(
    search: str = Query(default="", description="Search videos by title"),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    query = select(Video).order_by(Video.published_at.desc())
    count_query = select(func.count(Video.id))

    if search:
        query = query.where(Video.title.ilike(f"%{search}%"))
        count_query = count_query.where(Video.title.ilike(f"%{search}%"))

    total = (await db.execute(count_query)).scalar() or 0
    result = await db.execute(query.offset(skip).limit(limit))
    videos = result.scalars().all()

    return VideoListResponse(
        videos=[VideoResponse.model_validate(v) for v in videos],
        total=total,
    )


@router.get("/status", response_model=IngestionStatusResponse)
async def get_ingestion_status(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = IngestionService()
    status = await service.get_status()
    return IngestionStatusResponse(**status)


@router.post("/ingest", response_model=IngestionTriggerResponse)
async def trigger_ingestion(
    admin: User = Depends(require_admin),
):
    service = IngestionService()

    async def run_ingestion():
        try:
            processed = await service.ingest_channel()
            logger.info(f"Ingestion complete: {processed} videos processed")
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")

    asyncio.create_task(run_ingestion())

    status = await service.get_status()
    return IngestionTriggerResponse(
        message="Ingestion started in background",
        videos_found=status["total_videos"],
    )
