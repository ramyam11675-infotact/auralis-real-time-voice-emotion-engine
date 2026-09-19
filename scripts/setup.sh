#!/usr/bin/env bash
set -euo pipefail

echo "=== Auralis setup ==="

# 1. Check Ollama is installed / running
if ! command -v ollama &> /dev/null; then
  echo "Ollama not found. Install it from https://ollama.com/download"
  exit 1
fi

echo "Pulling llama3 model into local Ollama (this may take a few minutes)..."
ollama pull llama3

# 2. Reference voice check for zero-shot cloning
ASSET_DIR="$(dirname "$0")/../assets"
mkdir -p "$ASSET_DIR"
if [ ! -f "$ASSET_DIR/reference_voice.wav" ]; then
  echo ""
  echo "!! No reference_voice.wav found in $ASSET_DIR"
  echo "   Drop a clean ~10-20s mono WAV of the target voice there before"
  echo "   running the TTS engine (used for XTTSv2 zero-shot cloning)."
fi

# 3. Warn about GPU vs CPU
python3 - <<'PY'
import torch
if torch.cuda.is_available():
    print(f"GPU detected: {torch.cuda.get_device_name(0)} — set whisper_device/emotion_device to 'cuda' in config.py for best latency.")
else:
    print("No GPU detected — running on CPU. Expect higher latency; use whisper 'tiny.en' or 'base.en' for the Week-1 <200ms target.")
PY

echo "Setup complete. Start the backend with:"
echo "  uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload"
