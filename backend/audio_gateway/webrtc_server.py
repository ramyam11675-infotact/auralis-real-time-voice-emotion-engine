"""
WebRTC Audio Gateway (Week 1 foundation, Week 4 full-duplex).

Responsibilities:
  - Terminate the browser's WebRTC PeerConnection (UDP, not slow HTTP polling).
  - Pull incoming mic audio frames, convert to 16kHz float32, and hand them
    to the orchestrator's per-frame callback (VAD -> STT -> emotion).
  - Expose an outgoing MediaStreamTrack (`ResponseAudioTrack`) that the TTS
    streaming module pushes emotion-conditioned speech chunks into, frame
    by frame, in near-real-time.
  - Support instant interruption: `ResponseAudioTrack.clear()` empties the
    outgoing queue the instant the orchestrator detects user barge-in.
"""
from __future__ import annotations

import asyncio
import fractions
from typing import Callable, Optional

import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame
from loguru import logger

from backend.config import settings
from backend.utils.audio_utils import int16_to_float32, float32_to_int16, resample, to_mono


class ResponseAudioTrack(MediaStreamTrack):
    """
    Outgoing audio track that streams AI-generated speech back to the browser.
    The TTS module pushes PCM float32 chunks via `push_chunk()`; this track
    slices them into fixed WebRTC-frame-sized pieces on demand.

    Calling `clear()` immediately drops all queued/undelivered audio, which
    is how the Week-4 barge-in interruption is implemented at the transport
    layer (in addition to cancelling the TTS generation task upstream).
    """

    kind = "audio"

    def __init__(self, sample_rate: int = 48000):
        super().__init__()
        self.sample_rate = sample_rate
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self._pts = 0
        self._samples_per_frame = int(sample_rate * settings.frame_ms / 1000)
        self._silence = np.zeros(self._samples_per_frame, dtype=np.int16)
        self._leftover = np.zeros(0, dtype=np.int16)

    def push_chunk(self, pcm_float32: np.ndarray, source_sr: int):
        """Called by tts/streaming.py for every generated audio chunk."""
        pcm_resampled = resample(pcm_float32, source_sr, self.sample_rate)
        pcm_int16 = float32_to_int16(pcm_resampled)
        self._queue.put_nowait(pcm_int16)

    def clear(self):
        """Drop all pending audio — used on interruption/barge-in."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._leftover = np.zeros(0, dtype=np.int16)
        logger.debug("ResponseAudioTrack cleared (barge-in interruption).")

    async def recv(self) -> AudioFrame:
        # Assemble exactly one frame's worth of samples from leftover + queue,
        # falling back to silence if the TTS hasn't produced audio yet.
        while len(self._leftover) < self._samples_per_frame:
            try:
                chunk = await asyncio.wait_for(self._queue.get(), timeout=0.02)
                self._leftover = np.concatenate([self._leftover, chunk])
            except asyncio.TimeoutError:
                self._leftover = np.concatenate([self._leftover, self._silence])
                break

        frame_samples = self._leftover[: self._samples_per_frame]
        self._leftover = self._leftover[self._samples_per_frame :]

        frame = AudioFrame(format="s16", layout="mono", samples=len(frame_samples))
        frame.planes[0].update(frame_samples.tobytes())
        frame.sample_rate = self.sample_rate
        frame.pts = self._pts
        frame.time_base = fractions.Fraction(1, self.sample_rate)
        self._pts += len(frame_samples)
        return frame


class AuralisPeerConnection:
    """
    Wraps a single browser session's RTCPeerConnection: one incoming mic
    track, one outgoing response track, and the glue that feeds incoming
    audio frames to a caller-supplied async callback.
    """

    def __init__(self, on_incoming_frame: Callable[[np.ndarray], "asyncio.Future"]):
        self.pc = RTCPeerConnection()
        self.response_track = ResponseAudioTrack()
        self._on_incoming_frame = on_incoming_frame
        self._recv_task: Optional[asyncio.Task] = None

        self.pc.addTrack(self.response_track)

        @self.pc.on("track")
        def on_track(track: MediaStreamTrack):
            logger.info(f"Incoming track received: {track.kind}")
            if track.kind == "audio":
                self._recv_task = asyncio.create_task(self._consume_incoming(track))

        @self.pc.on("connectionstatechange")
        async def on_state_change():
            logger.info(f"WebRTC connection state: {self.pc.connectionState}")
            if self.pc.connectionState in ("failed", "closed"):
                await self.close()

    async def _consume_incoming(self, track: MediaStreamTrack):
        """Pull mic frames from the browser, normalize to 16kHz mono float32."""
        try:
            while True:
                frame: AudioFrame = await track.recv()
                pcm = frame.to_ndarray()
                if pcm.dtype == np.int16:
                    pcm = int16_to_float32(pcm)
                pcm = to_mono(pcm.astype(np.float32))
                pcm = resample(pcm, frame.sample_rate, settings.sample_rate)
                await self._on_incoming_frame(pcm)
        except MediaStreamError:
            logger.info("Incoming audio track ended.")
        except Exception:
            logger.exception("Incoming audio processing failed")

    async def handle_offer(self, sdp: str, type_: str) -> RTCSessionDescription:
        offer = RTCSessionDescription(sdp=sdp, type=type_)
        await self.pc.setRemoteDescription(offer)
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        return self.pc.localDescription

    async def close(self):
        if self._recv_task:
            self._recv_task.cancel()
        await self.pc.close()
