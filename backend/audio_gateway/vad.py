"""
Voice Activity Detection using Silero VAD.

Week 2 deliverable: detect exactly when the user stops speaking so the
orchestrator knows when to end STT accumulation and kick off the
LLM -> TTS turn. Also used to detect barge-in while the AI is speaking
(Week 4 full-duplex interruption handling).
"""
from __future__ import annotations

import time
import numpy as np
import torch
from loguru import logger

from backend.config import settings


class SileroVAD:
    """
    Thin async-friendly wrapper around torch.hub's Silero VAD model.

    Usage:
        vad = SileroVAD()
        for frame in audio_stream:               # frame: float32 np.array, 16kHz
            is_speech = vad.is_speech(frame)
            turn_ended = vad.update(is_speech)
            if turn_ended:
                ...  # trigger orchestrator turn
    """

    def __init__(self):
        logger.info("Loading Silero VAD model...")
        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=True,  # faster CPU inference
        )
        (self.get_speech_timestamps, *_rest) = utils

        self._speech_start_ts: float | None = None
        self._last_speech_ts: float | None = None
        self._in_speech = False

    def is_speech(self, frame: np.ndarray) -> float:
        """Return the raw speech probability [0, 1] for a single audio frame."""
        tensor = torch.from_numpy(frame).float()
        with torch.no_grad():
            prob = self.model(tensor, settings.sample_rate).item()
        return prob

    def update(self, frame: np.ndarray) -> dict:
        """
        Feed one frame, update internal speech/silence state machine, and
        return a dict describing what happened this frame:

            {
              "prob": float,
              "speaking": bool,          # currently inside a speech segment
              "turn_started": bool,      # speech just started (barge-in signal)
              "turn_ended": bool,        # enough trailing silence to end the turn
            }
        """
        now = time.monotonic()
        prob = self.is_speech(frame)
        is_speech_frame = prob >= settings.vad_threshold

        turn_started = False
        turn_ended = False

        if is_speech_frame:
            self._last_speech_ts = now
            if not self._in_speech:
                self._speech_start_ts = now
                self._in_speech = True
                turn_started = True
        else:
            if self._in_speech and self._last_speech_ts is not None:
                silence_ms = (now - self._last_speech_ts) * 1000
                if silence_ms >= settings.vad_min_silence_ms:
                    speech_duration_ms = (self._last_speech_ts - self._speech_start_ts) * 1000
                    if speech_duration_ms >= settings.vad_min_speech_ms:
                        turn_ended = True
                    self._in_speech = False
                    self._speech_start_ts = None

        return {
            "prob": prob,
            "speaking": self._in_speech,
            "turn_started": turn_started,
            "turn_ended": turn_ended,
        }

    def reset(self):
        self._speech_start_ts = None
        self._last_speech_ts = None
        self._in_speech = False
