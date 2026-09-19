"""
Per-session orchestrator: the state machine that ties VAD -> STT -> Emotion
-> LLM -> TTS together into one continuous, interruptible conversation.

One instance of `ConversationOrchestrator` is created per WebRTC connection
(see backend/main.py). It:

  1. Receives normalized 16kHz float32 frames from the WebRTC gateway.
  2. Feeds each frame to Silero VAD.
  3. While the user is speaking, buffers audio and streams partial
     transcripts + running emotion estimate to the frontend over the data
     channel (for live captions / waveform coloring).
  4. On VAD turn-end, runs a final STT pass + final emotion classification
     concurrently (asyncio.gather) to minimize latency.
  5. Kicks off the LLM streaming response, sentence-split, feeding XTTS
     streaming synthesis as sentences complete.
  6. If VAD detects the user speaking again *while* TTS is playing
     (barge-in), immediately cancels the TTS task and the in-flight LLM
     stream, clears the outgoing audio track, and returns to listening.
"""
from __future__ import annotations

import asyncio
from typing import Optional

import numpy as np
from loguru import logger

from backend.audio_gateway.vad import SileroVAD
from backend.audio_gateway.webrtc_server import ResponseAudioTrack
from backend.stt.whisper_engine import WhisperSTTEngine
from backend.stt.emotion_classifier import EmotionClassifier, EmotionResult
from backend.llm.context_engine import ContextEngine, split_into_sentences_stream
from backend.tts.xtts_engine import XTTSEngine
from backend.tts.streaming import StreamingTTSPlayer
from backend.utils.audio_utils import RollingPCMBuffer
from backend.config import settings


class ConversationOrchestrator:
    def __init__(
        self,
        response_track: ResponseAudioTrack,
        stt_engine: WhisperSTTEngine,
        emotion_engine: EmotionClassifier,
        llm_engine: ContextEngine,
        tts_engine: XTTSEngine,
        on_event=None,  # optional async callback(dict) -> None, for datachannel UI events
    ):
        self.vad = SileroVAD()
        self.stt = stt_engine
        self.emotion = emotion_engine
        self.llm = llm_engine
        self.tts_player = StreamingTTSPlayer(tts_engine, response_track)

        self._audio_buffer = RollingPCMBuffer()
        self._turn_buffer: list[np.ndarray] = []
        self._on_event = on_event
        self._turn_task: Optional[asyncio.Task] = None

    async def _emit(self, event: dict):
        if self._on_event:
            await self._on_event(event)

    async def handle_frame(self, frame: np.ndarray):
        """Called by the WebRTC gateway for every ~20ms incoming audio frame."""
        vad_state = self.vad.update(frame)

        # --- Barge-in: user started talking while AI is speaking ---
        if vad_state["turn_started"] and self.tts_player.is_speaking:
            logger.info("Barge-in detected — halting AI speech.")
            self.tts_player.stop()
            if self._turn_task and not self._turn_task.done():
                self._turn_task.cancel()
            await self._emit({"type": "interruption"})

        if vad_state["speaking"]:
            self._turn_buffer.append(frame)

        if vad_state["turn_ended"]:
            await self._on_turn_end()

    async def _on_turn_end(self):
        if not self._turn_buffer:
            return

        full_audio = np.concatenate(self._turn_buffer)
        self._turn_buffer = []
        self.vad.reset()

        self._turn_task = asyncio.create_task(self._run_turn(full_audio))

    async def _run_turn(self, audio: np.ndarray):
        try:
            # STT and emotion classification run concurrently — neither
            # blocks the other, which is what keeps end-to-end latency low.
            transcript_result, emotion_result = await asyncio.gather(
                asyncio.to_thread(self.stt.transcribe_chunk, audio, True),
                asyncio.to_thread(self.emotion.classify, audio),
            )

            transcript = transcript_result.text
            if not transcript:
                logger.debug("Empty transcript, skipping turn.")
                return

            await self._emit(
                {
                    "type": "user_turn",
                    "transcript": transcript,
                    "emotion": {
                        "arousal": emotion_result.arousal,
                        "valence": emotion_result.valence,
                        "dominance": emotion_result.dominance,
                        "label": emotion_result.label,
                    },
                    "stt_latency_ms": transcript_result.latency_ms,
                }
            )

            await self._respond(transcript, emotion_result)

        except asyncio.CancelledError:
            logger.info("Turn processing cancelled.")
        except Exception:
            logger.exception("Error while processing conversation turn")

    async def _respond(self, transcript: str, emotion: EmotionResult):
        text_stream = self.llm.stream_response(transcript, emotion)
        sentence_stream = split_into_sentences_stream(text_stream)

        # Emit each sentence to the frontend for live captioning as it's
        # generated, while simultaneously feeding it to TTS.
        async def _tee():
            async for sentence in sentence_stream:
                await self._emit({"type": "ai_sentence", "text": sentence})
                yield sentence

        tts_task = self.tts_player.start(_tee(), emotion)
        try:
            await tts_task
        except asyncio.CancelledError:
            pass
        finally:
            await self._emit({"type": "ai_turn_complete"})

    async def shutdown(self):
        self.tts_player.stop()
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
