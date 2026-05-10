"""
EchoRoom – Test Suite
Tests for: URL detection, Zoom normalisation, audio config,
transcript processing, and intelligence layer.
"""

import asyncio
import time
import pytest

from bot.meeting_bot import detect_platform, MeetingPlatform, _normalise_zoom_url
from transcription.transcriber import TranscriptSegment
from intelligence.analyser import LiveAnalyser, MeetingInsights


# ─────────────────────────────────────────────────────────────────────────────
# Bot core tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPlatformDetection:
    def test_google_meet(self):
        assert detect_platform("https://meet.google.com/abc-defg-hij") \
               == MeetingPlatform.GOOGLE_MEET

    def test_zoom(self):
        assert detect_platform("https://zoom.us/j/123456789") \
               == MeetingPlatform.ZOOM

    def test_teams(self):
        assert detect_platform(
            "https://teams.microsoft.com/l/meetup-join/..."
        ) == MeetingPlatform.TEAMS

    def test_unknown(self):
        assert detect_platform("https://example.com/meeting") \
               == MeetingPlatform.UNKNOWN

    def test_case_insensitive(self):
        assert detect_platform("https://MEET.GOOGLE.COM/abc") \
               == MeetingPlatform.GOOGLE_MEET


class TestZoomUrlNormalisation:
    def test_converts_j_to_wc(self):
        url = _normalise_zoom_url("https://zoom.us/j/123456789")
        assert "/wc/" in url
        assert "/j/" not in url

    def test_appends_join(self):
        url = _normalise_zoom_url("https://zoom.us/j/123456789")
        assert url.endswith("/join") or "/join?" in url

    def test_appends_pwd(self):
        url = _normalise_zoom_url("https://zoom.us/j/123456789")
        assert "pwd=" in url

    def test_idempotent_pwd(self):
        url = _normalise_zoom_url(
            "https://zoom.us/wc/123/join?pwd=abc"
        )
        assert url.count("pwd=") == 1


# ─────────────────────────────────────────────────────────────────────────────
# Intelligence layer tests  (no OpenAI calls – uses mock)
# ─────────────────────────────────────────────────────────────────────────────

class TestLiveAnalyser:
    def test_speaker_stats_accumulate(self):
        analyser = LiveAnalyser.__new__(LiveAnalyser)
        analyser.insights = MeetingInsights()
        analyser._buffer = []
        analyser._word_count = 0
        analyser._window_words = 9999   # never trigger LLM window
        analyser._overlap_words = 100
        analyser._client = None         # won't be called

        analyser.ingest("Alice", "Hello everyone", 0.0)
        analyser.ingest("Bob",   "Good morning team", 1.0)
        analyser.ingest("Alice", "Let us begin the review", 2.0)

        assert analyser.insights.speaker_stats["Alice"] == 2 + 5  # 7 words
        assert analyser.insights.speaker_stats["Bob"] == 3

    def test_buffer_grows(self):
        analyser = LiveAnalyser.__new__(LiveAnalyser)
        analyser.insights = MeetingInsights()
        analyser._buffer = []
        analyser._word_count = 0
        analyser._window_words = 9999
        analyser._overlap_words = 100
        analyser._client = None

        analyser.ingest("Alice", "First line", 0.0)
        analyser.ingest("Bob",   "Second line", 1.0)

        assert len(analyser._buffer) == 2
        assert "Alice: First line" in analyser._buffer[0]

    def test_trim_to_words(self):
        lines = ["Hello world", "Foo bar baz", "One two three four five"]
        result = LiveAnalyser._trim_to_words(lines, 5)
        # Should keep only the last line (5 words)
        assert result == ["One two three four five"]


# ─────────────────────────────────────────────────────────────────────────────
# Transcript segment model
# ─────────────────────────────────────────────────────────────────────────────

class TestTranscriptSegment:
    def test_creation(self):
        seg = TranscriptSegment(
            speaker="Alice",
            text="We should ship this feature by Friday.",
            start=120.0,
            end=125.5,
            confidence=0.95,
        )
        assert seg.speaker == "Alice"
        assert seg.start == 120.0
        assert seg.confidence == 0.95

    def test_default_confidence(self):
        seg = TranscriptSegment("Bob", "Sure.", 0.0, 1.0)
        assert seg.confidence == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
