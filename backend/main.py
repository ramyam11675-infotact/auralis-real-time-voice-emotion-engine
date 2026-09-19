"""
Auralis backend entrypoint.

- Loads all heavy ML models exactly once at startup (Whisper, emotion
  classifier, XTTS) via FastAPI's lifespan handler — never per-request.
- Exposes a WebRTC signaling endpoint (`POST /offer`) that the React
  frontend calls to establish the bi-directional audio PeerConnection.
- Wires each new PeerConnection to its own `ConversationOrchestrator`
  instance, and relays orchestrator events (transcripts, emotion, AI text)
  back to the browser over an RTCDataChannel for live UI updates.
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from backend.audio_gateway.webrtc_server import AuralisPeerConnection
from backend.stt.whisper_engine import WhisperSTTEngine
from backend.stt.emotion_classifier import EmotionClassifier
from backend.llm.context_engine import ContextEngine
from backend.tts.xtts_engine import XTTSEngine
from backend.pipeline.orchestrator import ConversationOrchestrator

# --- Global model handles (populated at startup) ---
models: dict = {}
sessions: dict[str, "Session"] = {}


class Session:
    def __init__(self, session_id: str, pc: AuralisPeerConnection, orchestrator: ConversationOrchestrator):
        self.id = session_id
        self.pc = pc
        self.orchestrator = orchestrator
        self.data_channel = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading models... this may take a while on first run.")
    models["stt"] = WhisperSTTEngine()
    models["emotion"] = EmotionClassifier()
    models["llm"] = ContextEngine()
    models["tts"] = XTTSEngine()
    logger.info("All models loaded. Auralis is ready.")
    yield
    logger.info("Shutting down: closing sessions...")
    for session in list(sessions.values()):
        await session.orchestrator.shutdown()
        await session.pc.close()
    await models["llm"].close()


app = FastAPI(title="Auralis", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": list(models.keys())}


@app.post("/offer")
async def offer(request: Request):
    """
    WebRTC signaling: receives the browser's SDP offer, creates a new
    PeerConnection + per-session orchestrator wired to the shared model
    instances, and returns the SDP answer.
    """
    body = await request.json()
    session_id = str(uuid.uuid4())

    async def on_orchestrator_event(event: dict):
        session = sessions.get(session_id)
        if session and session.data_channel and session.data_channel.readyState == "open":
            import json

            session.data_channel.send(json.dumps(event))

    async def on_incoming_frame(frame):
        session = sessions.get(session_id)
        if session:
            await session.orchestrator.handle_frame(frame)

    pc_wrapper = AuralisPeerConnection(on_incoming_frame=on_incoming_frame)

    orchestrator = ConversationOrchestrator(
        response_track=pc_wrapper.response_track,
        stt_engine=models["stt"],
        emotion_engine=models["emotion"],
        llm_engine=models["llm"],
        tts_engine=models["tts"],
        on_event=on_orchestrator_event,
    )

    session = Session(session_id, pc_wrapper, orchestrator)
    sessions[session_id] = session

    @pc_wrapper.pc.on("datachannel")
    def on_datachannel(channel):
        session.data_channel = channel
        logger.info(f"Data channel '{channel.label}' opened for session {session_id}")

    @pc_wrapper.pc.on("connectionstatechange")
    async def on_state_change():
        if pc_wrapper.pc.connectionState in ("failed", "closed"):
            sessions.pop(session_id, None)

    answer = await pc_wrapper.handle_offer(body["sdp"], body["type"])

    return JSONResponse(
        {
            "sdp": answer.sdp,
            "type": answer.type,
            "session_id": session_id,
        }
    )


@app.post("/session/{session_id}/close")
async def close_session(session_id: str):
    session = sessions.pop(session_id, None)
    if session:
        await session.orchestrator.shutdown()
        await session.pc.close()
    return {"closed": bool(session)}
