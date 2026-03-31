import logging
from typing import AsyncGenerator

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are TiiffBot, an AI assistant powered by the coaching content of Tiffany Cheng — an executive leadership coach who has held global director and VP roles at Volvo Group and Atlas Copco, and has helped leaders secure roles as CFOs, CTOs, VPs, GMs, and Senior Directors.

You answer questions about career advancement, leadership development, executive presence, promotion strategies, and business coaching based on Tiffany's YouTube video content.

IMPORTANT RULES:
1. Answer ONLY based on the provided video transcript context. Do not make up information.
2. If the context doesn't contain enough information to answer, say so honestly and suggest the user explore Tiffany's other videos or visit inspiremyday.org.
3. When referencing advice, mention which video it comes from.
4. Be encouraging, practical, and actionable — matching Tiffany's coaching style.
5. Use clear, concise language suitable for ambitious business professionals.
6. If asked about topics outside the video content, politely redirect to the available topics."""


def _build_messages(context_chunks: list[dict], question: str, chat_history: list[dict]) -> list[dict]:
    context_text = "\n\n---\n\n".join(
        f"[From: \"{chunk['title']}\" at {int(chunk['start_time'] // 60)}:{int(chunk['start_time'] % 60):02d}]\n{chunk['text']}"
        for chunk in context_chunks
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    user_content = f"""Based on the following video transcript excerpts, please answer the user's question.

VIDEO CONTEXT:
{context_text}

USER QUESTION: {question}"""

    messages.append({"role": "user", "content": user_content})
    return messages


async def stream_openai(
    context_chunks: list[dict],
    question: str,
    chat_history: list[dict],
) -> AsyncGenerator[str, None]:
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    messages = _build_messages(context_chunks, question, chat_history)

    stream = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        stream=True,
        temperature=0.7,
        max_tokens=1500,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


async def stream_anthropic(
    context_chunks: list[dict],
    question: str,
    chat_history: list[dict],
) -> AsyncGenerator[str, None]:
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    messages = _build_messages(context_chunks, question, chat_history)

    system_msg = messages[0]["content"]
    chat_msgs = messages[1:]

    async with client.messages.stream(
        model="claude-sonnet-4-20250514",
        system=system_msg,
        messages=chat_msgs,
        max_tokens=1500,
        temperature=0.7,
    ) as stream:
        async for text in stream.text_stream:
            yield text


async def stream_response(
    provider: str,
    context_chunks: list[dict],
    question: str,
    chat_history: list[dict],
) -> AsyncGenerator[str, None]:
    if provider == "anthropic":
        async for token in stream_anthropic(context_chunks, question, chat_history):
            yield token
    else:
        async for token in stream_openai(context_chunks, question, chat_history):
            yield token
