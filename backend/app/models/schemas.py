from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr


# --- Auth ---

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: str
    email: str
    name: Optional[str]
    is_admin: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# --- Chat ---

class ChatMessageCreate(BaseModel):
    content: str
    provider: Optional[str] = "openai"  # "openai" or "anthropic"


class ChatMessageResponse(BaseModel):
    id: str
    role: str
    content: str
    sources: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatSessionResponse(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ChatSessionDetailResponse(BaseModel):
    id: str
    title: str
    messages: list[ChatMessageResponse]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# --- Videos ---

class VideoResponse(BaseModel):
    id: str
    youtube_id: str
    title: str
    description: Optional[str]
    thumbnail_url: Optional[str]
    channel_title: Optional[str]
    published_at: Optional[str]
    duration: Optional[str]
    transcript_status: str
    chunk_count: int
    created_at: datetime

    model_config = {"from_attributes": True}


class VideoListResponse(BaseModel):
    videos: list[VideoResponse]
    total: int


# --- Ingestion ---

class IngestionStatusResponse(BaseModel):
    total_videos: int
    completed: int
    failed: int
    pending: int
    total_chunks: int


class IngestionTriggerResponse(BaseModel):
    message: str
    videos_found: int


# --- Chat sources ---

class VideoSource(BaseModel):
    video_id: str
    youtube_id: str
    title: str
    thumbnail_url: Optional[str]
    timestamp_seconds: Optional[float]
    relevance_score: float
