"""
Shared low-level audio helpers used by STT, emotion, VAD, and TTS modules.

Everything internally is float32 mono PCM at `settings.sample_rate` (16kHz)
unless explicitly noted. WebRTC/Opus frames coming from aiortc are int16;
TTS output from XTTS is float32 at 24kHz and must be resampled down to the
WebRTC track's expected rate before sending.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

from backend.config import settings


def int16_to_float32(pcm: np.ndarray) -> np.ndarray:
    """Convert int16 PCM samples to float32 in [-1, 1]."""
    return (pcm.astype(np.float32)) / 32768.0


def float32_to_int16(pcm: np.ndarray) -> np.ndarray:
    """Convert float32 PCM in [-1, 1] to int16, with clipping."""
    clipped = np.clip(pcm, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def resample(pcm: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample float32 mono PCM from orig_sr to target_sr."""
    if orig_sr == target_sr:
        return pcm
    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    return resample_poly(pcm, up, down).astype(np.float32)


def to_mono(pcm: np.ndarray) -> np.ndarray:
    """Collapse a (channels, samples) or (samples, channels) array to mono."""
    if pcm.ndim == 1:
        return pcm
    return pcm.mean(axis=0 if pcm.shape[0] < pcm.shape[1] else 1).astype(np.float32)


class RollingPCMBuffer:
    """
    Accumulates streamed int16/float32 audio frames and yields fixed-size
    chunks (e.g. every `settings.stt_chunk_seconds`) for the STT engine,
    without blocking the WebRTC receive loop.
    """

    def __init__(self, chunk_seconds: float | None = None, sample_rate: int | None = None):
        self.sample_rate = sample_rate or settings.sample_rate
        self.chunk_size = int((chunk_seconds or settings.stt_chunk_seconds) * self.sample_rate)
        self._buf = np.zeros(0, dtype=np.float32)

    def push(self, frame: np.ndarray) -> list[np.ndarray]:
        """Append a frame; return list of complete chunks now available."""
        self._buf = np.concatenate([self._buf, frame])
        chunks = []
        while len(self._buf) >= self.chunk_size:
            chunks.append(self._buf[: self.chunk_size])
            self._buf = self._buf[self.chunk_size :]
        return chunks

    def flush(self) -> np.ndarray | None:
        """Return and clear any remaining partial buffer (e.g. on end-of-turn)."""
        if len(self._buf) == 0:
            return None
        remainder = self._buf
        self._buf = np.zeros(0, dtype=np.float32)
        return remainder

    def clear(self):
        self._buf = np.zeros(0, dtype=np.float32)
