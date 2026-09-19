"""
Mid-project review latency harness.

Measures, using a sample WAV file (simulating a trainee's spoken sentence):
  1. STT latency          — time to transcribe the buffered chunk (target < 200ms)
  2. TTFT                 — end-of-speech -> first LLM token (target < 800ms)
  3. End-to-end latency   — end-of-speech -> first audio byte streamed back (target < 800ms)

Usage:
    python scripts/test_latency.py path/to/sample_utterance.wav
"""
from __future__ import annotations

import asyncio
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from backend.config import settings
from backend.stt.whisper_engine import WhisperSTTEngine
from backend.stt.emotion_classifier import EmotionClassifier
from backend.llm.context_engine import ContextEngine
from backend.tts.xtts_engine import XTTSEngine


async def main(wav_path: str):
    print(f"Loading models... (device: stt={settings.whisper_device}, emotion={settings.emotion_device})")
    stt = WhisperSTTEngine()
    emotion_clf = EmotionClassifier()
    llm = ContextEngine()
    tts = XTTSEngine()

    audio, sr = sf.read(wav_path, dtype="float32")
    if sr != settings.sample_rate:
        raise ValueError(f"Sample rate mismatch: expected {settings.sample_rate}, got {sr}")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    print("\n--- 1. STT latency ---")
    t0 = time.perf_counter()
    result = stt.transcribe_chunk(audio, is_final=True)
    stt_ms = (time.perf_counter() - t0) * 1000
    print(f"Transcript: {result.text!r}")
    print(f"STT latency: {stt_ms:.0f}ms (target < {settings.target_stt_latency_ms}ms) "
          f"{'PASS' if stt_ms < settings.target_stt_latency_ms else 'FAIL'}")

    print("\n--- Emotion classification ---")
    t0 = time.perf_counter()
    emotion = emotion_clf.classify(audio)
    emo_ms = (time.perf_counter() - t0) * 1000
    print(f"Emotion: arousal={emotion.arousal:.2f} valence={emotion.valence:.2f} "
          f"label={emotion.label} ({emo_ms:.0f}ms)")

    print("\n--- 2. Time-To-First-Token (LLM) ---")
    t_end_of_speech = time.perf_counter()
    first_token_time = None
    full_text = ""
    async for delta in llm.stream_response(result.text, emotion):
        if first_token_time is None:
            first_token_time = time.perf_counter()
        full_text += delta
    ttft_ms = (first_token_time - t_end_of_speech) * 1000
    print(f"LLM response: {full_text!r}")
    print(f"TTFT: {ttft_ms:.0f}ms (target < {settings.target_ttft_ms}ms) "
          f"{'PASS' if ttft_ms < settings.target_ttft_ms else 'FAIL'}")

    print("\n--- 3. End-to-end (speech end -> first TTS audio byte) ---")
    t0 = time.perf_counter()
    first_chunk = None
    for chunk in tts.synthesize_stream(full_text.split(".")[0] + ".", emotion):
        first_chunk = chunk
        break
    e2e_ms = (time.perf_counter() - t_end_of_speech) * 1000
    print(f"First audio chunk shape: {None if first_chunk is None else first_chunk.shape}")
    print(f"End-to-end latency: {e2e_ms:.0f}ms (target < {settings.target_e2e_latency_ms}ms) "
          f"{'PASS' if e2e_ms < settings.target_e2e_latency_ms else 'FAIL'}")

    await llm.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/test_latency.py <path_to_wav>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
