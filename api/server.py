"""
EchoRoom – REST API & WebSocket Server
FastAPI app exposing:
  POST /sessions          – start a new bot session
  GET  /sessions/{id}     – session status + live insights
  POST /sessions/{id}/ask – Q&A endpoint
  POST /sessions/{id}/stop – gracefully stop a session
  WS   /ws/{id}           – real-time transcript & insight stream
"""

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from bot.meeting_bot import BotConfig, MeetingBot
from intelligence.analyser import LiveAnalyser, MeetingInsights, MeetingQA, ReportGenerator
from transcription.audio_capture import AudioCapture, AudioConfig
from transcription.transcriber import TranscriptionConfig, WhisperTranscriber

logger = logging.getLogger("echoroom.api")
logging.basicConfig(level=logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

class SessionState:
    def __init__(self, session_id: str, meeting_url: str):
        self.id = session_id
        self.meeting_url = meeting_url
        self.status = "initialising"
        self.start_time = time.time()
        self.transcript_lines: list[str] = []
        self.bot: Optional[MeetingBot] = None
        self.audio: Optional[AudioCapture] = None
        self.transcriber: Optional[WhisperTranscriber] = None
        self.analyser: Optional[LiveAnalyser] = None
        self.qa: Optional[MeetingQA] = None
        self.ws_clients: list[WebSocket] = []
        self.insights = MeetingInsights()


_sessions: Dict[str, SessionState] = {}


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Cleanup all sessions on shutdown
    for s in list(_sessions.values()):
        await _stop_session(s)

app = FastAPI(title="EchoRoom API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class StartSessionRequest(BaseModel):
    meeting_url: str
    bot_name: str = "EchoRoom Bot"
    whisper_model: str = "base"
    diarize: bool = False
    google_email: Optional[str] = None
    google_password: Optional[str] = None


class AskRequest(BaseModel):
    question: str


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/sessions", status_code=201)
async def start_session(req: StartSessionRequest):
    session_id = str(uuid.uuid4())
    state = SessionState(session_id, req.meeting_url)
    _sessions[session_id] = state

    asyncio.create_task(_run_session(state, req))
    return {"session_id": session_id, "status": "initialising"}


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    state = _get_or_404(session_id)
    return {
        "session_id": state.id,
        "status": state.status,
        "meeting_url": state.meeting_url,
        "uptime_s": round(time.time() - state.start_time),
        "transcript_lines": len(state.transcript_lines),
        "insights": _serialise_insights(state.insights),
    }


@app.post("/sessions/{session_id}/ask")
async def ask_question(session_id: str, req: AskRequest):
    state = _get_or_404(session_id)
    if state.qa is None:
        # Index on first ask
        if not state.transcript_lines:
            raise HTTPException(422, "No transcript yet.")
        state.qa = MeetingQA()
        state.qa.index(state.transcript_lines)
    answer = state.qa.ask(req.question)
    return {"question": req.question, "answer": answer}


@app.post("/sessions/{session_id}/stop")
async def stop_session(session_id: str):
    state = _get_or_404(session_id)
    await _stop_session(state)
    return {"session_id": session_id, "status": "stopped"}


@app.get("/sessions/{session_id}/report")
async def get_report(session_id: str):
    state = _get_or_404(session_id)
    gen = ReportGenerator()
    report = gen.generate(state.insights, "\n".join(state.transcript_lines))
    return {"report": report}


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    state = _sessions.get(session_id)
    if not state:
        await websocket.close(code=4404)
        return
    await websocket.accept()
    state.ws_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()   # keep connection alive
    except WebSocketDisconnect:
        state.ws_clients.remove(websocket)


# ─────────────────────────────────────────────────────────────────────────────
# Session orchestration
# ─────────────────────────────────────────────────────────────────────────────

async def _run_session(state: SessionState, req: StartSessionRequest) -> None:
    try:
        state.status = "joining"

        # 1. Launch bot
        bot_cfg = BotConfig(
            bot_name=req.bot_name,
            headless=True,
            google_account_email=req.google_email,
            google_account_password=req.google_password,
        )
        state.bot = MeetingBot(bot_cfg)
        await state.bot.join(req.meeting_url)

        # 2. Set up analyser
        state.analyser = LiveAnalyser()

        # 3. Set up transcriber
        t_cfg = TranscriptionConfig(
            whisper_model=req.whisper_model,
            diarize=req.diarize,
        )
        state.transcriber = WhisperTranscriber(
            on_segment=lambda seg: asyncio.run_coroutine_threadsafe(
                _on_segment(state, seg), asyncio.get_event_loop()
            ),
            config=t_cfg,
        )
        state.transcriber.load()
        state.transcriber.start(time.time())

        # 4. Set up audio capture → feeds transcriber
        state.audio = AudioCapture(
            on_chunk=state.transcriber.submit_chunk,
            config=AudioConfig(),
        )
        state.audio.start()

        # 5. Also drain the bot's caption queue (DOM-scraped captions)
        asyncio.create_task(_drain_caption_queue(state))

        state.status = "active"
        logger.info("Session %s is active.", state.id)

    except Exception as exc:
        state.status = f"error: {exc}"
        logger.error("Session %s failed: %s", state.id, exc)


async def _on_segment(state: SessionState, seg) -> None:
    """Called for each transcribed segment; update state + broadcast."""
    line = f"{seg.speaker}: {seg.text}"
    state.transcript_lines.append(line)
    state.analyser.ingest(seg.speaker, seg.text, seg.start)
    state.insights = state.analyser.insights

    payload = {
        "type": "segment",
        "speaker": seg.speaker,
        "text": seg.text,
        "timestamp": seg.start,
        "insights": _serialise_insights(state.insights),
    }
    await _broadcast(state, payload)


async def _drain_caption_queue(state: SessionState) -> None:
    """Forward DOM-scraped captions as transcript segments."""
    while state.status == "active":
        try:
            entry = await asyncio.wait_for(
                state.bot.caption_queue.get(), timeout=1.0
            )
            speaker = entry.get("speaker", "Unknown")
            text = entry.get("text", "")
            if text:
                line = f"{speaker} (caption): {text}"
                state.transcript_lines.append(line)
                state.analyser.ingest(speaker, text, entry["timestamp"])
                state.insights = state.analyser.insights
                await _broadcast(state, {
                    "type": "caption",
                    "speaker": speaker,
                    "text": text,
                    "timestamp": entry["timestamp"],
                })
        except asyncio.TimeoutError:
            continue
        except Exception:
            break


async def _broadcast(state: SessionState, payload: dict) -> None:
    import json
    dead = []
    for ws in state.ws_clients:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        state.ws_clients.remove(ws)


async def _stop_session(state: SessionState) -> None:
    state.status = "stopping"
    if state.audio:
        state.audio.stop()
    if state.transcriber:
        state.transcriber.stop()
    if state.bot:
        await state.bot.leave()
    state.analyser.finalise()
    state.status = "stopped"
    logger.info("Session %s stopped.", state.id)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_or_404(session_id: str) -> SessionState:
    state = _sessions.get(session_id)
    if not state:
        raise HTTPException(404, f"Session {session_id} not found.")
    return state


def _serialise_insights(ins: MeetingInsights) -> dict:
    return {
        "summary": ins.summary,
        "key_decisions": ins.key_decisions,
        "action_items": [
            {"owner": a.owner, "task": a.task,
             "due": a.due, "priority": a.priority}
            for a in ins.action_items
        ],
        "topics": ins.topics,
        "speaker_stats": ins.speaker_stats,
        "sentiment_flags": [
            {"speaker": s.speaker, "sentiment": s.sentiment,
             "score": s.score, "text": s.text}
            for s in ins.sentiment_flags[-10:]   # last 10
        ],
    }
