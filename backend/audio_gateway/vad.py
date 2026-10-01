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
    Silero VAD wrapper that buffers incoming audio frames to the
    512-sample window required by the 16kHz Silero model.
    """

    def __init__(self):
        logger.info("Loading Silero VAD model...")

        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=True,
        )

        (self.get_speech_timestamps, *_rest) = utils

        self._speech_start_ts: float | None = None
        self._last_speech_ts: float | None = None
        self._in_speech = False

        # Silero VAD requires exactly 512 samples at 16kHz.
        self._vad_frame_size = 512
        self._vad_buffer = np.zeros(0, dtype=np.float32)

    def is_speech(self, frame: np.ndarray) -> float:
        """Return the raw speech probability [0, 1]."""

        tensor = torch.from_numpy(frame).float()

        with torch.no_grad():
            prob = self.model(tensor, settings.sample_rate).item()

        return prob

    def update(self, frame: np.ndarray) -> dict:
        """
        Feed incoming audio into the VAD.

        WebRTC may provide 640-sample frames, while Silero requires
        512 samples at 16kHz. Buffer incoming samples and process
        complete 512-sample windows.
        """

        self._vad_buffer = np.concatenate(
            [self._vad_buffer, frame.astype(np.float32)]
        )

        probabilities = []

        while len(self._vad_buffer) >= self._vad_frame_size:
            vad_frame = self._vad_buffer[:self._vad_frame_size]
            self._vad_buffer = self._vad_buffer[self._vad_frame_size:]

            probabilities.append(self.is_speech(vad_frame))

        # Not enough audio for one Silero window yet.
        if not probabilities:
            return {
                "prob": 0.0,
                "speaking": self._in_speech,
                "turn_started": False,
                "turn_ended": False,
            }

        prob = max(probabilities)
        is_speech_frame = prob >= settings.vad_threshold

        now = time.monotonic()

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
                    speech_duration_ms = (
                        self._last_speech_ts - self._speech_start_ts
                    ) * 1000

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
        self._vad_buffer = np.zeros(0, dtype=np.float32)