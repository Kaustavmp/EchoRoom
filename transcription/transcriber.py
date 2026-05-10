"""
EchoRoom – Transcription & Speaker Diarization Layer
Uses OpenAI Whisper for ASR and pyannote.audio for speaker diarization.
Falls back gracefully if pyannote is unavailable.
"""

import logging
import os
import queue
import threading
from dataclasses import dataclass, field
from typing import Callable, List, Optional

logger = logging.getLogger("echoroom.transcription")


@dataclass
class TranscriptSegment:
    speaker: str
    text: str
    start: float   # seconds from meeting start
    end: float
    confidence: float = 1.0


@dataclass
class TranscriptionConfig:
    whisper_model: str = "base"        # tiny/base/small/medium/large
    device: str = "cpu"               # "cpu" or "cuda"
    language: Optional[str] = None    # None = auto-detect
    diarize: bool = False             # requires pyannote + HF token
    hf_token: Optional[str] = None   # Hugging Face token for pyannote
    min_speakers: int = 1
    max_speakers: int = 10


class WhisperTranscriber:
    """
    Wraps OpenAI Whisper for chunk-based transcription.
    Thread-safe via an internal processing queue.
    """

    def __init__(
        self,
        on_segment: Callable[[TranscriptSegment], None],
        config: TranscriptionConfig | None = None,
    ):
        self.on_segment = on_segment
        self.cfg = config or TranscriptionConfig()
        self._model = None
        self._diarizer = None
        self._queue: queue.Queue[str] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._meeting_start: float = 0.0

    def load(self) -> None:
        """Load Whisper (and optionally pyannote) models – call once at startup."""
        import whisper  # type: ignore
        logger.info("Loading Whisper model: %s on %s",
                    self.cfg.whisper_model, self.cfg.device)
        self._model = whisper.load_model(
            self.cfg.whisper_model, device=self.cfg.device
        )
        logger.info("Whisper model loaded.")

        if self.cfg.diarize:
            self._load_diarizer()

    def _load_diarizer(self) -> None:
        try:
            from pyannote.audio import Pipeline  # type: ignore
            token = self.cfg.hf_token or os.environ.get("HF_TOKEN")
            if not token:
                logger.warning(
                    "diarize=True but HF_TOKEN not set. "
                    "Diarization disabled."
                )
                self.cfg.diarize = False
                return
            self._diarizer = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=token,
            )
            logger.info("Pyannote speaker diarization loaded.")
        except ImportError:
            logger.warning(
                "pyannote.audio not installed. "
                "Install: pip install pyannote.audio  "
                "Diarization disabled."
            )
            self.cfg.diarize = False

    def start(self, meeting_start: float) -> None:
        self._meeting_start = meeting_start
        self._running = True
        self._thread = threading.Thread(
            target=self._processing_loop, daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._queue.put(None)   # sentinel
        if self._thread:
            self._thread.join(timeout=30)

    def submit_chunk(self, wav_path: str) -> None:
        """Called by AudioCapture for each completed WAV chunk."""
        self._queue.put(wav_path)

    def transcribe_file(self, wav_path: str) -> List[TranscriptSegment]:
        """Synchronous transcription of a single WAV file."""
        if self._model is None:
            raise RuntimeError("Call .load() before transcribing.")
        return self._run_transcription(wav_path)

    # ------------------------------------------------------------------ #
    #  Internal                                                             #
    # ------------------------------------------------------------------ #

    def _processing_loop(self) -> None:
        while self._running:
            path = self._queue.get()
            if path is None:
                break
            try:
                for seg in self._run_transcription(path):
                    self.on_segment(seg)
            except Exception as exc:
                logger.error("Transcription error for %s: %s", path, exc)

    def _run_transcription(self, wav_path: str) -> List[TranscriptSegment]:
        import whisper  # type: ignore

        result = self._model.transcribe(
            wav_path,
            language=self.cfg.language,
            word_timestamps=False,
            verbose=False,
        )

        raw_segments = result.get("segments", [])

        if self.cfg.diarize and self._diarizer and raw_segments:
            return self._merge_diarization(wav_path, raw_segments)

        # No diarization – return segments with "Speaker 1"
        out = []
        for seg in raw_segments:
            out.append(TranscriptSegment(
                speaker="Speaker 1",
                text=seg["text"].strip(),
                start=self._meeting_start + seg["start"],
                end=self._meeting_start + seg["end"],
            ))
        return out

    def _merge_diarization(self, wav_path: str, raw_segments: list) -> List[TranscriptSegment]:
        """Align Whisper segments with pyannote speaker turns."""
        import torch  # type: ignore

        diarization = self._diarizer(
            wav_path,
            min_speakers=self.cfg.min_speakers,
            max_speakers=self.cfg.max_speakers,
        )

        # Build speaker timeline: list of (start, end, speaker_label)
        speaker_turns = [
            (turn.start, turn.end, label)
            for turn, _, label in diarization.itertracks(yield_label=True)
        ]

        out = []
        for seg in raw_segments:
            mid = (seg["start"] + seg["end"]) / 2
            speaker = self._find_speaker(mid, speaker_turns)
            out.append(TranscriptSegment(
                speaker=speaker,
                text=seg["text"].strip(),
                start=self._meeting_start + seg["start"],
                end=self._meeting_start + seg["end"],
            ))
        return out

    @staticmethod
    def _find_speaker(t: float, turns: list) -> str:
        """Return the speaker label active at time t."""
        for start, end, label in turns:
            if start <= t <= end:
                return label
        # Nearest turn fallback
        if turns:
            closest = min(turns, key=lambda x: abs(x[0] - t))
            return closest[2]
        return "Unknown"
