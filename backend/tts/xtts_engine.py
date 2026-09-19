"""
Zero-Shot Voice Cloning + emotion-conditioned speech generation via XTTSv2
(Coqui TTS).

Week 3 deliverable: condition generation on an emotional baseline (calm vs
panicked) and stream audio in chunks rather than waiting for the whole
sentence.

XTTSv2 doesn't take an explicit "emotion" parameter like some models, so we
approximate emotional prosody with the two levers it actually exposes:
  - `speed`      : higher arousal -> slightly faster speech
  - `temperature`: higher arousal/lower valence -> more prosodic variance
      (a bit more expressive/less monotone, at the cost of stability)
A calmer target additionally gets a mild reduction to both, for a steadier,
de-escalating tone — matching the crisis-negotiation use case where the AI
mirrors and gradually lowers the intensity of the room.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger
from TTS.api import TTS

from backend.config import settings
from backend.stt.emotion_classifier import EmotionResult


@dataclass
class TTSGenerationParams:
    speed: float
    temperature: float


def emotion_to_tts_params(emotion: EmotionResult) -> TTSGenerationParams:
    """Map arousal/valence into XTTS generation knobs."""
    # arousal 0..1 -> speed 0.85..1.25
    speed = 0.85 + emotion.arousal * 0.40
    # low valence (negative) and high arousal both push temperature up
    # (more erratic/urgent-sounding delivery); calm+positive stays low/stable.
    temperature = 0.55 + emotion.arousal * 0.25 + (1 - emotion.valence) * 0.15
    temperature = min(temperature, 0.95)
    return TTSGenerationParams(speed=round(speed, 2), temperature=round(temperature, 2))


class XTTSEngine:
    def __init__(self):
        logger.info(f"Loading XTTS model '{settings.tts_model_name}'...")
        self.tts = TTS(settings.tts_model_name)
        self.output_sample_rate = 24000  # XTTSv2 native output rate

    def synthesize_stream(self, text: str, emotion: EmotionResult):
        """
        Yields float32 PCM chunks at `self.output_sample_rate` as XTTS
        generates them (Coqui's `inference_stream` yields per-token audio
        frames rather than returning a single completed waveform).

        The caller (tts/streaming.py) pushes each chunk onto the WebRTC
        ResponseAudioTrack immediately, so playback starts before the full
        sentence has finished generating.
        """
        params = emotion_to_tts_params(emotion)
        logger.debug(
            f"Synthesizing (speed={params.speed}, temperature={params.temperature}): "
            f"{text!r}"
        )

        gpt_cond_latent, speaker_embedding = self.tts.synthesizer.tts_model.get_conditioning_latents(
            audio_path=[settings.tts_reference_voice]
        )

        chunk_stream = self.tts.synthesizer.tts_model.inference_stream(
            text,
            settings.tts_language,
            gpt_cond_latent,
            speaker_embedding,
            speed=params.speed,
            temperature=params.temperature,
            enable_text_splitting=True,
        )

        for chunk in chunk_stream:
            pcm = chunk.cpu().numpy().astype(np.float32).squeeze()
            yield pcm
