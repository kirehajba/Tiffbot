# TiiffBot - AI Business Coach

An AI-powered business coaching assistant that answers questions based on Tiffany Cheng's YouTube video content. Built with a RAG (Retrieval-Augmented Generation) architecture that ingests video transcripts and uses them as context for AI responses.

**Channel**: [@inspiremydaytiffany](https://www.youtube.com/@inspiremydaytiffany)

## Architecture

- **Frontend**: Next.js 14 (App Router) + Tailwind CSS
- **Backend**: Python FastAPI
- **Vector DB**: ChromaDB (transcript embeddings)
- **Database**: PostgreSQL (users, chat history, video metadata)
- **LLM Support**: OpenAI GPT-4o + Anthropic Claude
- **Embeddings**: OpenAI text-embedding-3-small

## Quick Start

### Prerequisites

- Docker & Docker Compose
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))
- Anthropic API key (optional, [get one here](https://console.anthropic.com/))

No YouTube API key is needed -- the app scrapes your channel directly.

### 1. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and add your **OpenAI API key** (required). The YouTube channel is already configured to `@inspiremydaytiffany`. Generate JWT/NextAuth secrets with:

```bash
openssl rand -hex 32
```

### 2. Run with Docker Compose

```bash
docker compose up --build
```

This starts all services:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- PostgreSQL: localhost:5432
- ChromaDB: localhost:8100

### 3. Create an admin user

Sign up at http://localhost:3000/signup, then promote yourself to admin:

```bash
docker compose exec postgres psql -U tiiffbot -d tiiffbot -c \
  "UPDATE users SET is_admin = true WHERE email = 'your@email.com';"
```

### 4. Ingest your videos

Go to http://localhost:3000/admin and click **Start Ingestion**. This will:
1. Fetch all videos from your YouTube channel (no API key needed)
2. Extract transcripts (auto-generated or manual captions)
3. Chunk and embed the transcripts using OpenAI
4. Store them in ChromaDB for semantic search

### 5. Start chatting

Go to http://localhost:3000/chat and ask business coaching questions. The AI will answer using your video content and cite the source videos with timestamps.

## Local Development (without Docker)

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Make sure PostgreSQL and ChromaDB are running (via docker compose up postgres chromadb)
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Project Structure

```
TiiffBot/
├── frontend/               # Next.js web app
│   ├── src/
│   │   ├── app/            # Pages (login, signup, chat, videos, admin)
│   │   ├── components/     # Sidebar, ChatMessage, ChatInput, VideoCard
│   │   └── lib/            # API client, auth context, utilities
│   └── package.json
├── backend/                # Python FastAPI
│   ├── app/
│   │   ├── api/            # Auth, chat, videos endpoints
│   │   ├── services/       # YouTube, ingestion, RAG, LLM services
│   │   ├── models/         # SQLAlchemy models + Pydantic schemas
│   │   └── core/           # Config, dependency injection
│   └── requirements.txt
├── docker-compose.yml
├── .env.example
└── README.md
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/auth/signup | Create account |
| POST | /api/auth/login | Sign in |
| GET | /api/auth/me | Get current user |
| POST | /api/chat/sessions | Create chat session |
| GET | /api/chat/sessions | List chat sessions |
| GET | /api/chat/sessions/:id | Get session with messages |
| POST | /api/chat/sessions/:id/messages | Send message (SSE streaming) |
| DELETE | /api/chat/sessions/:id | Delete session |
| GET | /api/videos | List indexed videos |
| GET | /api/videos/status | Get ingestion status |
| POST | /api/videos/ingest | Trigger ingestion (admin) |

## How It Works

1. **Ingestion**: Videos are scraped from your YouTube channel (no API key needed). Transcripts are extracted, chunked into ~500-token segments with timestamps, embedded using OpenAI, and stored in ChromaDB.

2. **Chat (RAG)**: When a user asks a question, the query is embedded and used to find the top 5 most relevant transcript chunks. These chunks are sent as context to the LLM, which generates an answer grounded in your video content.

3. **Sources**: Each answer includes links to the source videos with timestamps, so users can watch the relevant sections directly.
