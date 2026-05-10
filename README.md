# EchoRoom v2 🎙

**Intelligent Meeting Bot — Joins, Listens, Understands.**

EchoRoom automatically joins Google Meet, Zoom, and Microsoft Teams meetings,
captures the audio, transcribes it in real time using OpenAI Whisper, identifies
who is speaking, and surfaces live insights — summaries, action items, sentiment
flags, and decisions — through a beautiful dashboard and REST API.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        EchoRoom v2                              │
│                                                                 │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │  Layer 1: Bot    │    │  Layer 2: Audio  │                   │
│  │  (Joining)       │    │  (Listening)     │                   │
│  │                  │    │                  │                   │
│  │  Playwright      │───▶│  ffmpeg loopback │                   │
│  │  Stealth mode    │    │  Whisper ASR     │                   │
│  │  Google Meet ✓   │    │  pyannote        │                   │
│  │  Zoom Web ✓      │    │  (diarization)   │                   │
│  │  MS Teams ✓      │    │                  │                   │
│  │  DOM captions ✓  │    │  WAV chunks →    │                   │
│  └──────────────────┘    │  TranscriptSegs  │                   │
│          │               └────────┬─────────┘                   │
│          │                        │                             │
│          └────────────┬───────────┘                             │
│                       ▼                                         │
│           ┌───────────────────────┐                             │
│           │  Layer 3: Intelligence│                             │
│           │                       │                             │
│           │  LiveAnalyser (GPT-4) │                             │
│           │  Rolling window       │                             │
│           │  → Summary            │                             │
│           │  → Action items       │                             │
│           │  → Sentiment flags    │                             │
│           │  → Key decisions      │                             │
│           │                       │                             │
│           │  MeetingQA (RAG)      │                             │
│           │  FAISS + sentence-    │                             │
│           │  transformers         │                             │
│           └───────────┬───────────┘                             │
│                       │                                         │
│           ┌───────────▼───────────┐                             │
│           │  Layer 4: API +       │                             │
│           │  Dashboard            │                             │
│           │                       │                             │
│           │  FastAPI REST         │                             │
│           │  WebSocket stream     │                             │
│           │  HTML Dashboard       │                             │
│           └───────────────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/yourorg/echoroom.git
cd echoroom

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY
```

### 3a. CLI — join a meeting directly

```bash
python echoroom.py join "https://meet.google.com/abc-def-ghi" \
    --name "EchoRoom Bot" \
    --model base \
    --save meeting_2025
```

### 3b. API server + dashboard

```bash
# Terminal 1 – API
python echoroom.py server

# Browser – open the dashboard
open dashboard/index.html
```

### 3c. Docker (production)

```bash
docker compose up -d
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/sessions` | Start a new bot session |
| `GET` | `/sessions/{id}` | Get session status + insights |
| `POST` | `/sessions/{id}/ask` | Ask a question about the meeting |
| `GET` | `/sessions/{id}/report` | Generate full Markdown report |
| `POST` | `/sessions/{id}/stop` | Stop the session |
| `WS` | `/ws/{id}` | Real-time transcript + insight stream |

### Start a session

```json
POST /sessions
{
  "meeting_url": "https://meet.google.com/abc-def-ghi",
  "bot_name": "EchoRoom Bot",
  "whisper_model": "base",
  "diarize": false
}
```

### WebSocket payload (per segment)

```json
{
  "type": "segment",
  "speaker": "Speaker 1",
  "text": "We should ship this by Friday.",
  "timestamp": 1718000000.0,
  "insights": {
    "summary": "...",
    "action_items": [...],
    "key_decisions": [...],
    "topics": [...],
    "sentiment_flags": [...]
  }
}
```

---

## Project Structure

```
EchoRoom_v2/
├── bot/
│   └── meeting_bot.py        # Playwright bot — joins Meet/Zoom/Teams
├── transcription/
│   ├── audio_capture.py      # ffmpeg loopback audio capture
│   └── transcriber.py        # Whisper ASR + pyannote diarization
├── intelligence/
│   └── analyser.py           # Live LLM analysis, RAG Q&A, report gen
├── api/
│   └── server.py             # FastAPI REST + WebSocket server
├── dashboard/
│   └── index.html            # Real-time HTML dashboard
├── tests/
│   └── test_echoroom.py      # Pytest suite
├── docker/
│   ├── entrypoint.sh         # PulseAudio + uvicorn startup
│   └── pulse-default.pa      # PulseAudio config
├── echoroom.py               # CLI entry point
├── Dockerfile                # Multi-stage Docker build
├── docker-compose.yml        # Full stack: API + Redis + Nginx
├── requirements.txt          # All Python deps
└── .env.example              # Environment variable template
```

---

## Feature Matrix

| Feature | Original | v2 |
|---------|----------|-----|
| Selenium → Playwright | ✗ | ✅ |
| Stealth / anti-detection | ✗ | ✅ |
| Google Meet support | partial | ✅ |
| Zoom web client | ✗ | ✅ |
| Microsoft Teams | ✗ | ✅ |
| DOM caption scraping | ✗ | ✅ |
| Loopback audio (Linux/Mac/Win) | basic | ✅ |
| Whisper transcription | ✅ | ✅ |
| Speaker diarization | ✗ | ✅ (pyannote) |
| Live LLM analysis | ✗ | ✅ |
| Action item extraction | ✗ | ✅ |
| Sentiment flagging | ✗ | ✅ |
| RAG Q&A over transcript | basic | ✅ (FAISS) |
| Full meeting report | ✗ | ✅ (Markdown) |
| FastAPI REST server | ✗ | ✅ |
| WebSocket streaming | ✗ | ✅ |
| Real-time dashboard | ✗ | ✅ |
| Docker + Compose | ✗ | ✅ |
| Test suite | ✗ | ✅ |
| CLI tool | ✗ | ✅ |
| Hardcoded paths removed | ✗ | ✅ |

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | required | OpenAI API key |
| `HF_TOKEN` | optional | Hugging Face token for pyannote diarization |
| `ECHOROOM_WHISPER_MODEL` | `base` | ASR model size |
| `ECHOROOM_LIVE_MODEL` | `gpt-4o-mini` | LLM for live analysis |
| `ECHOROOM_REPORT_MODEL` | `gpt-4o` | LLM for final report |
| `GOOGLE_EMAIL` | optional | Google account to sign in the bot |
| `GOOGLE_PASSWORD` | optional | Google account password |
| `REDIS_URL` | optional | Redis for multi-worker session sharing |

---

## Infrastructure & Scaling

Each bot instance requires:
- **~1 GB RAM** (Chromium browser)
- **~0.5 CPU core** (idle), ~2 cores during Whisper inference
- **Persistent audio sink** (PulseAudio virtual sink in Docker)

For high scale:
- Run one bot per Docker container
- Use a load balancer with sticky sessions (session_id → container)
- Store session state in Redis (REDIS_URL env var)
- Use `whisper_model=tiny` or offload to Whisper API for lower latency

### Alternative: Recall.ai

If building the joining logic is too complex for your timeline, consider
[Recall.ai](https://www.recall.ai) — a managed bot-joining API.
Replace `bot/meeting_bot.py` with a single Recall API call and EchoRoom's
intelligence layer still works identically with the returned transcript.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## License

MIT
