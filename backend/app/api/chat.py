import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sse_starlette.sse import EventSourceResponse

from app.api.auth import get_current_user
from app.core.deps import get_db
from app.models.database import ChatMessage, ChatSession, User, async_session_factory
from app.models.schemas import (
    ChatMessageCreate,
    ChatSessionDetailResponse,
    ChatSessionResponse,
)
from app.services.rag import RAGService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/sessions", response_model=ChatSessionResponse)
async def create_session(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    session = ChatSession(user_id=user.id)
    db.add(session)
    await db.flush()
    await db.refresh(session)
    return ChatSessionResponse.model_validate(session)


@router.get("/sessions", response_model=list[ChatSessionResponse])
async def list_sessions(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.user_id == user.id)
        .order_by(ChatSession.updated_at.desc())
    )
    sessions = result.scalars().all()
    return [ChatSessionResponse.model_validate(s) for s in sessions]


@router.get("/sessions/{session_id}", response_model=ChatSessionDetailResponse)
async def get_session(
    session_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.id == session_id, ChatSession.user_id == user.id)
        .options(selectinload(ChatSession.messages))
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return ChatSessionDetailResponse.model_validate(session)


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == user.id)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    await db.delete(session)
    return {"detail": "Session deleted"}


@router.post("/sessions/{session_id}/messages")
async def send_message(
    session_id: str,
    data: ChatMessageCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.id == session_id, ChatSession.user_id == user.id)
        .options(selectinload(ChatSession.messages))
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    user_message = ChatMessage(
        session_id=session_id,
        role="user",
        content=data.content,
    )
    db.add(user_message)

    if session.title == "New Chat":
        session.title = data.content[:80]

    await db.commit()

    chat_history = [
        {"role": msg.role, "content": msg.content}
        for msg in sorted(session.messages, key=lambda m: m.created_at)
    ]
    chat_history.append({"role": "user", "content": data.content})

    rag = RAGService()
    provider = data.provider or "openai"

    async def event_generator():
        full_response = ""

        try:
            token_stream, sources = await rag.query_stream(
                question=data.content,
                chat_history=chat_history,
                provider=provider,
            )

            async for token in token_stream:
                full_response += token
                yield {"event": "token", "data": token}

            yield {
                "event": "sources",
                "data": json.dumps([
                    {
                        "video_id": s["video_id"],
                        "youtube_id": s["youtube_id"],
                        "title": s["title"],
                        "thumbnail_url": s["thumbnail_url"],
                        "timestamp_seconds": s["timestamp_seconds"],
                        "relevance_score": s["relevance_score"],
                    }
                    for s in sources
                ]),
            }

            async with async_session_factory() as save_db:
                assistant_message = ChatMessage(
                    session_id=session_id,
                    role="assistant",
                    content=full_response,
                    sources=json.dumps(sources) if sources else None,
                )
                save_db.add(assistant_message)
                await save_db.commit()

            yield {"event": "done", "data": ""}

        except Exception as e:
            logger.error(f"Error in chat stream: {e}")
            yield {"event": "error", "data": str(e)}

    return EventSourceResponse(event_generator())
