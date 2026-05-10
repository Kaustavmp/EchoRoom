# ── EchoRoom Docker Image ──────────────────────────────────────────────────
# Multi-stage build: keeps the final image lean by separating build deps.
# The container includes a full Chromium (for Playwright) + ffmpeg + Python.
#
# Build:  docker build -t echoroom:latest .
# Run:    docker run -p 8000:8000 --env-file .env echoroom:latest
# ──────────────────────────────────────────────────────────────────────────

# ── Stage 1: Python deps ──────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile wheels (e.g. numpy, pyannote)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="EchoRoom <echoroom@yourorg.com>"
LABEL version="2.0.0"

# Runtime system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    # ffmpeg for audio capture
    ffmpeg \
    # Chromium dependencies (for Playwright)
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    # PulseAudio virtual sink for loopback audio in headless env
    pulseaudio \
    pulseaudio-utils \
    # Fonts for Google Meet / Zoom UI
    fonts-liberation \
    fonts-noto-color-emoji \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Install Playwright browsers (Chromium only)
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN pip install playwright \
    && playwright install chromium \
    && playwright install-deps chromium

# Copy application code
COPY . .

# PulseAudio configuration for headless loopback capture
COPY docker/pulse-default.pa /etc/pulse/default.pa

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

# Entrypoint: start PulseAudio, then the API server
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
