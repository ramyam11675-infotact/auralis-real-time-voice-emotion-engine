"""
Acoustic emotion extraction using a Wav2Vec2 model fine-tuned for
continuous arousal/valence/dominance regression (not just discrete labels).

This runs *in parallel* with Whisper transcription on the same audio chunk
(the orchestrator fires both as concurrent asyncio tasks), which is what
lets Auralis "hear" panic/anger/sadness in addition to just reading words.

Model: audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
Outputs three values in [0, 1]:
    arousal    - calm (0) <-> excited/panicked (1)
    dominance  - submissive (0) <-> in-control (1)
    valence    - negative (0) <-> positive (1)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2PreTrainedModel, Wav2Vec2Model
from loguru import logger

from backend.config import settings


@dataclass
class EmotionResult:
    arousal: float
    dominance: float
    valence: float
    label: str  # human-readable bucket derived from arousal/valence

    def to_prompt_context(self) -> str:
        """Serialize for injection into the LLM persona prompt (see llm/prompts.py)."""
        return (
            f"[USER EMOTIONAL STATE] arousal={self.arousal:.2f} "
            f"valence={self.valence:.2f} dominance={self.dominance:.2f} "
            f"(perceived as: {self.label})"
        )


class _RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, 3)  # arousal, dominance, valence

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


class _EmotionModel(Wav2Vec2PreTrainedModel):
    """HF-style wrapper matching the audeering checkpoint's architecture."""

    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = _RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        pooled = hidden_states.mean(dim=1)
        return self.classifier(pooled)


def _bucket_label(arousal: float, valence: float) -> str:
    """Map continuous (arousal, valence) into a coarse label useful for logging/UI."""
    if arousal > 0.65 and valence < 0.4:
        return "panicked/distressed"
    if arousal > 0.65 and valence >= 0.4:
        return "agitated/excited"
    if arousal <= 0.65 and valence < 0.35:
        return "sad/withdrawn"
    if arousal <= 0.35 and valence >= 0.5:
        return "calm"
    return "neutral"


class EmotionClassifier:
    def __init__(self):
        logger.info(f"Loading emotion model '{settings.emotion_model_id}'...")
        self.processor = Wav2Vec2Processor.from_pretrained(settings.emotion_model_id)
        self.model = _EmotionModel.from_pretrained(settings.emotion_model_id)
        self.model.to(settings.emotion_device)
        self.model.eval()

    @torch.no_grad()
    def classify(self, pcm_float32: np.ndarray) -> EmotionResult:
        inputs = self.processor(
            pcm_float32, sampling_rate=settings.sample_rate, return_tensors="pt"
        )
        input_values = inputs.input_values.to(settings.emotion_device)
        logits = self.model(input_values)
        arousal, dominance, valence = logits[0].clamp(0, 1).cpu().tolist()

        return EmotionResult(
            arousal=arousal,
            dominance=dominance,
            valence=valence,
            label=_bucket_label(arousal, valence),
        )
