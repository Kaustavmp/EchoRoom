"""
EchoRoom – Audio Capture Layer
Captures system/loopback audio in a cross-platform way and writes
WAV chunks for the transcription pipeline.
"""

import asyncio
import logging
import os
import platform
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("echoroom.audio")


@dataclass
class AudioConfig:
    sample_rate: int = 16_000          # 16 kHz – optimal for Whisper
    channels: int = 1                  # mono
    chunk_duration_s: int = 30         # each WAV chunk length
    output_dir: str = "audio_chunks"
    ffmpeg_bin: str = "ffmpeg"

    # Override auto-detection by setting these:
    force_device: Optional[str] = None
    force_format: Optional[str] = None


def _detect_loopback_device(cfg: AudioConfig) -> tuple[str, str]:
    """Return (format_flag, device) for the current OS loopback source."""
    if cfg.force_format and cfg.force_device:
        return cfg.force_format, cfg.force_device

    system = platform.system()

    if system == "Linux":
        # PulseAudio monitor source – captures what the speakers play
        monitor = _pulse_get_monitor_source()
        return "pulse", monitor or "default.monitor"

    if system == "Darwin":
        # BlackHole or Soundflower must be installed; fall back to :0
        bh = _macos_find_blackhole()
        return "avfoundation", f":{bh}" if bh else ":0"

    if system == "Windows":
        # Stereo Mix or WASAPI loopback
        sm = _windows_find_stereo_mix()
        if sm:
            return "dshow", f"audio={sm}"
        # Fallback: WASAPI loopback (any output device)
        return "dshow", "audio=virtual-audio-capturer"

    raise RuntimeError(f"Unsupported OS: {system}")


def _pulse_get_monitor_source() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["pactl", "list", "short", "sources"], text=True
        )
        for line in out.splitlines():
            if ".monitor" in line:
                return line.split()[1]
    except Exception:
        pass
    return None


def _macos_find_blackhole() -> Optional[int]:
    try:
        out = subprocess.check_output(
            ["ffmpeg", "-f", "avfoundation", "-list_devices",
             "true", "-i", "\"\""],
            stderr=subprocess.STDOUT, text=True
        )
        for line in out.splitlines():
            if "BlackHole" in line or "Soundflower" in line:
                import re
                m = re.search(r"\[(\d+)\]", line)
                if m:
                    return int(m.group(1))
    except Exception:
        pass
    return None


def _windows_find_stereo_mix() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            stderr=subprocess.STDOUT, text=True, errors="replace"
        )
        for line in out.splitlines():
            if "Stereo Mix" in line or "stereo mix" in line.lower():
                import re
                m = re.search(r'"([^"]+)"', line)
                if m:
                    return m.group(1)
    except Exception:
        pass
    return None


class AudioCapture:
    """
    Captures loopback audio in overlapping chunks and calls
    on_chunk(wav_path) for each completed chunk.
    """

    def __init__(
        self,
        on_chunk: Callable[[str], None],
        config: AudioConfig | None = None,
    ):
        self.on_chunk = on_chunk
        self.cfg = config or AudioConfig()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        if not shutil.which(self.cfg.ffmpeg_bin):
            raise RuntimeError(
                "ffmpeg not found. Install it: https://ffmpeg.org/download.html"
            )
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("Audio capture started.")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("Audio capture stopped.")

    def _capture_loop(self) -> None:
        fmt, device = _detect_loopback_device(self.cfg)
        chunk_idx = 0

        while not self._stop_event.is_set():
            out_path = os.path.join(
                self.cfg.output_dir, f"chunk_{chunk_idx:05d}.wav"
            )
            cmd = [
                self.cfg.ffmpeg_bin,
                "-y",                                  # overwrite
                "-f", fmt,
                "-i", device,
                "-t", str(self.cfg.chunk_duration_s),  # chunk length
                "-ar", str(self.cfg.sample_rate),
                "-ac", str(self.cfg.channels),
                "-acodec", "pcm_s16le",
                out_path,
            ]
            try:
                subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=self.cfg.chunk_duration_s + 10,
                )
                if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                    logger.debug("Audio chunk ready: %s", out_path)
                    self.on_chunk(out_path)
                chunk_idx += 1
            except subprocess.TimeoutExpired:
                logger.warning("ffmpeg chunk timed out, restarting.")
            except Exception as exc:
                logger.error("Audio capture error: %s", exc)
                self._stop_event.wait(2)
