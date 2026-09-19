"""
Chunked audio streaming glue: takes text sentence-by-sentence from the LLM,
runs each through XTTS's streaming inference, and pushes resulting PCM
chunks onto the WebRTC ResponseAudioTrack in near-real-time — all inside a
single cancellable asyncio.Task so Week-4 interruption handling can kill it
instantly mid-sentence.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator

from loguru import logger

from backend.audio_gateway.webrtc_server import ResponseAudioTrack
from backend.stt.emotion_classifier import EmotionResult
from backend.tts.xtts_engine import XTTSEngine


class StreamingTTSPlayer:
    """
    Wraps one AI "turn": consumes a stream of complete sentences from the LLM
    (see llm.context_engine.split_into_sentences_stream) and synthesizes +
    streams each one to the browser, sentence after sentence, without
    waiting for the full LLM response to finish first.
    """

    def __init__(self, engine: XTTSEngine, track: ResponseAudioTrack):
        self.engine = engine
        self.track = track
        self._task: asyncio.Task | None = None

    async def _run(self, sentence_stream: AsyncIterator[str], emotion: EmotionResult):
        try:
            async for sentence in sentence_stream:
                # Run the (CPU/GPU-bound) synthesis generator in a thread so
                # it doesn't block the asyncio event loop that's also
                # servicing the WebRTC connection.
                loop = asyncio.get_running_loop()

                def _produce_chunks():
                    return list(self.engine.synthesize_stream(sentence, emotion))

                pcm_chunks = await loop.run_in_executor(None, _produce_chunks)

                for chunk in pcm_chunks:
                    self.track.push_chunk(chunk, source_sr=self.engine.output_sample_rate)
                    # Yield control so cancellation can interleave between chunks.
                    await asyncio.sleep(0)
        except asyncio.CancelledError:
            logger.info("TTS playback cancelled (barge-in).")
            raise

    def start(self, sentence_stream: AsyncIterator[str], emotion: EmotionResult):
        """Begin streaming a new AI turn. Cancels any turn already in flight."""
        self.stop()
        self._task = asyncio.create_task(self._run(sentence_stream, emotion))
        return self._task

    def stop(self):
        """
        Immediately halt TTS generation and clear any audio already queued
        for playback — this is the Week-4 "instantly halt and listen" path,
        triggered the moment VAD detects the user has started talking again.
        """
        if self._task and not self._task.done():
            self._task.cancel()
        self.track.clear()

    @property
    def is_speaking(self) -> bool:
        return self._task is not None and not self._task.done()
