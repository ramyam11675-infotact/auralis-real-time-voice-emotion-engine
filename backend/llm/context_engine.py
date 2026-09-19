"""
Context Engine — streams tokens from a local Llama 3 model served by Ollama.

Week 2 deliverable: connect transcribed text to a pre-prompted persona.
Mid-project deliverable: measure Time-To-First-Token (TTFT) from end-of-user
-speech to the first streamed token (see scripts/test_latency.py).

Design note: this yields tokens as an async generator so the TTS streaming
module (Week 3) can start synthesizing speech for the first sentence while
the LLM is still generating the rest — sentence-level pipelining instead of
waiting for the full response.
"""
from __future__ import annotations

import time
from typing import AsyncIterator

import httpx
from loguru import logger

from backend.config import settings
from backend.llm.prompts import build_system_prompt, build_user_turn
from backend.stt.emotion_classifier import EmotionResult


class ContextEngine:
    def __init__(self):
        self._client = httpx.AsyncClient(base_url=settings.ollama_host, timeout=30.0)
        self._history: list[dict] = []

    async def stream_response(
        self, transcript: str, emotion: EmotionResult
    ) -> AsyncIterator[str]:
        """
        Streams response text token-by-token (actually: small text deltas as
        Ollama emits them). Raises asyncio.CancelledError cleanly if the
        caller cancels this generator on barge-in — the httpx stream is
        closed in the `finally` block.
        """
        system_prompt = build_system_prompt(settings.llm_persona, emotion)
        user_message = build_user_turn(transcript)

        messages = [{"role": "system", "content": system_prompt}]
        messages += self._history
        messages.append({"role": "user", "content": user_message})

        start = time.perf_counter()
        first_token_logged = False
        full_response = ""

        try:
            async with self._client.stream(
                "POST",
                "/api/chat",
                json={
                    "model": settings.llm_model,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": settings.llm_temperature,
                        "num_predict": settings.llm_max_tokens,
                    },
                },
            ) as response:
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    import json

                    payload = json.loads(line)
                    delta = payload.get("message", {}).get("content", "")

                    if delta and not first_token_logged:
                        ttft_ms = (time.perf_counter() - start) * 1000
                        first_token_logged = True
                        if ttft_ms > settings.target_ttft_ms:
                            logger.warning(f"TTFT {ttft_ms:.0f}ms exceeded target")
                        else:
                            logger.info(f"TTFT: {ttft_ms:.0f}ms")

                    if delta:
                        full_response += delta
                        yield delta

                    if payload.get("done"):
                        break
        finally:
            if full_response:
                self._history.append({"role": "user", "content": user_message})
                self._history.append({"role": "assistant", "content": full_response})
                # Keep history bounded so prompt size (and latency) stays flat.
                self._history = self._history[-12:]

    def reset_history(self):
        self._history = []

    async def close(self):
        await self._client.aclose()


def split_into_sentences_stream(text_deltas: AsyncIterator[str]):
    """
    Async generator that buffers streamed text deltas and yields complete
    sentences as soon as they're detected (on '.', '!', '?', or newline),
    so tts/streaming.py can start synthesizing each sentence immediately
    instead of waiting for the entire LLM response.
    """
    buffer = ""
    terminators = (".", "!", "?", "\n")

    async def _gen():
        nonlocal buffer
        async for delta in text_deltas:
            buffer += delta
            while True:
                cut = -1
                for i, ch in enumerate(buffer):
                    if ch in terminators:
                        cut = i
                        break
                if cut == -1:
                    break
                sentence = buffer[: cut + 1].strip()
                buffer = buffer[cut + 1 :]
                if sentence:
                    yield sentence
        if buffer.strip():
            yield buffer.strip()

    return _gen()
