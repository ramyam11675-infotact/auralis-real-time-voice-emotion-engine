# Auralis — Real-Time Voice-to-Voice Emotion Engine

A low-latency, emotion-aware voice conversation engine for crisis-negotiation training simulators.

Instead of the classic slow pipeline (STT → LLM → TTS run sequentially, each waiting on the last),
Auralis runs everything as a **streaming, concurrent pipeline**:

```
 Mic (WebRTC/UDP)
       │
       ▼
 ┌───────────────┐     ┌────────────────────┐
 │  VAD (Silero) │────▶│ Faster-Whisper STT  │──▶ transcript chunks
 └───────────────┘     └──────────┬──────────┘
       │                          │
       │              ┌───────────▼───────────┐
       │              │ Wav2Vec2 Emotion Model │──▶ {arousal, valence, label}
       │              └───────────┬───────────┘
       │                          │
       │              ┌───────────▼───────────┐
       │              │  Llama 3 (Ollama) w/  │──▶ streamed response tokens
       │              │  emotion-injected     │
       │              │  persona prompt        │
       │              └───────────┬───────────┘
       │                          │
       │              ┌───────────▼───────────┐
       │              │  XTTSv2 streaming TTS  │──▶ 20ms PCM chunks
       │              │  (emotion-conditioned) │
       │              └───────────┬───────────┘
       │                          │
       ▼                          ▼
 ┌─────────────────────────────────────────┐
 │      aiortc WebRTC Audio Gateway         │──▶ speaker (browser)
 └─────────────────────────────────────────┘
```

Full-duplex: if the user starts speaking while the AI is talking (barge-in), the
orchestrator cancels the in-flight TTS stream immediately (`asyncio.Task.cancel()`)
and returns control to the STT/VAD path.

## Project layout

```
auralis/
├── backend/
│   ├── main.py                     # FastAPI + aiortc signaling server
│   ├── config.py                   # central settings
│   ├── audio_gateway/
│   │   ├── webrtc_server.py        # aiortc PeerConnection + audio tracks
│   │   └── vad.py                  # Silero VAD wrapper
│   ├── stt/
│   │   ├── whisper_engine.py       # faster-whisper streaming transcription
│   │   └── emotion_classifier.py   # Wav2Vec2 arousal/valence classifier
│   ├── llm/
│   │   ├── context_engine.py       # Ollama/Llama3 streaming client
│   │   └── prompts.py              # persona + emotion-injection prompt templates
│   ├── tts/
│   │   ├── xtts_engine.py          # XTTSv2 zero-shot, emotion-conditioned TTS
│   │   └── streaming.py            # chunked PCM streaming + barge-in cancellation
│   ├── pipeline/
│   │   └── orchestrator.py         # glues STT→Emotion→LLM→TTS together per-turn
│   └── utils/
│       └── audio_utils.py          # resampling, framing, WAV helpers
├── frontend/                       # React + Vite app
│   └── src/
│       ├── App.jsx
│       ├── hooks/useWebRTC.js
│       └── components/{AudioRecorder,WaveformVisualizer,ConversationLog}.jsx
├── scripts/
│   ├── setup.sh                    # installs deps, pulls Ollama model
│   └── test_latency.py             # measures TTFB / TTFT per week's mid-project review
├── requirements.txt
└── docker-compose.yml
```

## Quick start

```bash
# 1. Backend
cd auralis
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/setup.sh          # pulls llama3 into local Ollama
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# 2. Frontend (separate terminal)
cd auralis/frontend
npm install
npm run dev
```

Open the printed Vite URL, click **Connect**, allow microphone access, and start talking.

## Week-by-week mapping to this codebase

| Week | Deliverable | Files |
|---|---|---|
| 1 | Whisper STT < 200ms, WebRTC bidirectional audio | `stt/whisper_engine.py`, `audio_gateway/webrtc_server.py`, `frontend/` |
| 2 | Llama3 persona integration, Silero VAD | `llm/context_engine.py`, `llm/prompts.py`, `audio_gateway/vad.py` |
| Mid | Transcription audit + TTFT latency check | `scripts/test_latency.py` |
| 3 | Emotion-conditioned XTTS, chunked streaming | `tts/xtts_engine.py`, `tts/streaming.py` |
| 4 | Barge-in / full-duplex, waveform visualizer | `pipeline/orchestrator.py` (`cancel_current_turn`), `frontend/src/components/WaveformVisualizer.jsx` |

## Notes on running models locally

- **STT**: `faster-whisper` with `small.en` or `base.en` + `int8` quantization gets you
  sub-200ms partial transcripts on a decent GPU, ~400-600ms on CPU.
- **Emotion**: `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` outputs continuous
  arousal/valence/dominance — good for a controllable emotional axis, not just discrete
  labels.
- **LLM**: run `ollama pull llama3` and `ollama serve` (default `http://localhost:11434`).
- **TTS**: `coqui-tts` (XTTSv2) needs a one-time reference `.wav` per voice for zero-shot
  cloning, and supports streaming inference via `model.inference_stream(...)`.
- All heavy model loads happen once at process startup in `backend/main.py`'s lifespan
  handler, not per-request.
