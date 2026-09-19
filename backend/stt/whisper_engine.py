"""
Real-time Speech-to-Text using faster-whisper (CTranslate2 backend).

Week 1 target: transcribe rolling ~1s audio chunks in under 200ms using
`small.en`/`int8` on CPU, or `float16` on GPU. We run in "streaming partial"
mode: every buffered chunk is transcribed independently for low-latency
partial results, and the orchestrator concatenates + reconciles partials
into a final transcript once VAD signals turn-end.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel
from loguru import logger

from backend.config import settings


@dataclass
class TranscriptResult:
    text: str
    latency_ms: float
    is_final: bool
    avg_logprob: float


class WhisperSTTEngine:
    def __init__(self):
        logger.info(
            f"Loading Whisper model '{settings.whisper_model}' "
            f"on {settings.whisper_device} ({settings.whisper_compute_type})..."
        )
        self.model = WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
        self._running_text: list[str] = []

    def transcribe_chunk(self, pcm_float32: np.ndarray, is_final: bool = False) -> TranscriptResult:
        """
        Transcribe a single chunk of 16kHz mono float32 audio.
        Called by the orchestrator each time RollingPCMBuffer yields a chunk,
        and once more on turn-end with the flushed remainder (is_final=True).
        """
        start = time.perf_counter()

        segments, info = self.model.transcribe(
            pcm_float32,
            language="en",
            beam_size=settings.whisper_beam_size,
            vad_filter=False,  # VAD is handled upstream by Silero for turn-taking
            condition_on_previous_text=True,
            word_timestamps=False,
        )

        text_parts = []
        avg_logprob = 0.0
        n = 0
        for seg in segments:
            text_parts.append(seg.text.strip())
            avg_logprob += seg.avg_logprob
            n += 1

        text = " ".join(text_parts).strip()
        latency_ms = (time.perf_counter() - start) * 1000

        if latency_ms > settings.target_stt_latency_ms:
            logger.warning(
                f"STT latency {latency_ms:.0f}ms exceeded target "
                f"{settings.target_stt_latency_ms}ms for chunk of "
                f"{len(pcm_float32) / settings.sample_rate:.2f}s"
            )

        if text:
            self._running_text.append(text)

        return TranscriptResult(
            text=text,
            latency_ms=latency_ms,
            is_final=is_final,
            avg_logprob=(avg_logprob / n) if n else 0.0,
        )

    def get_full_transcript(self) -> str:
        return " ".join(self._running_text).strip()

    def reset(self):
        self._running_text = []
