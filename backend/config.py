"""
Central configuration for Auralis.

All tunables live here so latency budgets (Week-1/Mid-project targets) can be
adjusted without hunting through the pipeline code.
"""
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # --- General ---
    app_name: str = "Auralis"
    sample_rate: int = 16000          # Whisper + Wav2Vec2 expect 16kHz mono
    frame_ms: int = 20                # WebRTC Opus frame size in ms
    audio_channels: int = 1

    # --- STT (Week 1 target: < 200ms partial transcript) ---
    whisper_model: str = "small.en"   # tiny.en/base.en for even lower latency
    whisper_device: str = "cpu"       # "cuda" if GPU available
    whisper_compute_type: str = "int8"  # int8 quantization for CPU speed
    whisper_beam_size: int = 1        # greedy decoding = fastest
    stt_chunk_seconds: float = 1.0    # audio buffer flushed to Whisper every N sec

    # --- VAD ---
    vad_threshold: float = 0.5
    vad_min_silence_ms: int = 500     # silence duration that ends a turn
    vad_min_speech_ms: int = 250      # minimum speech duration to count as "talking"

    # --- Emotion classifier ---
    emotion_model_id: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    emotion_device: str = "cpu"

    # --- LLM (Context Engine) ---
    ollama_host: str = "http://localhost:11434"
    llm_model: str = "llama3"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 200
    llm_persona: str = "crisis_negotiator"

    # --- TTS ---
    tts_model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    tts_reference_voice: str = str(Path(__file__).parent.parent / "assets" / "reference_voice.wav")
    tts_language: str = "en"
    tts_chunk_size_ms: int = 200      # size of each streamed audio chunk sent over WebRTC

    # --- Latency budgets (used by scripts/test_latency.py) ---
    target_stt_latency_ms: int = 200
    target_ttft_ms: int = 800         # end-of-user-speech -> first LLM token
    target_e2e_latency_ms: int = 800  # end-of-user-speech -> first audio byte back

    class Config:
        env_file = ".env"
        env_prefix = "AURALIS_"


settings = Settings()
