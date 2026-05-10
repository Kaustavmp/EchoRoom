#!/bin/bash
# EchoRoom Docker Entrypoint
# Sets up a PulseAudio virtual audio sink so ffmpeg can capture
# the browser's audio output in a headless environment.

set -e

# ── PulseAudio virtual sink ───────────────────────────────────────────────
echo "Starting PulseAudio..."
pulseaudio --start --exit-idle-time=-1 --daemon

sleep 1

# Create a null sink (virtual speaker) – Chromium will output here
pactl load-module module-null-sink sink_name=EchoRoomSink \
    sink_properties=device.description="EchoRoom-Virtual-Sink"

# Create a loopback from the sink's monitor to the default source
pactl load-module module-loopback source=EchoRoomSink.monitor

# Set as default source so ffmpeg picks it up automatically
pactl set-default-source EchoRoomSink.monitor
pactl set-default-sink EchoRoomSink

echo "PulseAudio virtual sink ready."

# ── Start FastAPI ─────────────────────────────────────────────────────────
echo "Starting EchoRoom API server..."
exec uvicorn api.server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --loop asyncio \
    --log-level info
