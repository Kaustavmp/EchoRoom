#!/usr/bin/env python3
"""
EchoRoom CLI – Quick Start
Run a meeting bot from the command line without the full API server.

Usage:
    python echoroom.py join <url> [--name "My Bot"] [--model base]
    python echoroom.py transcribe <audio.wav>
    python echoroom.py qa <transcript.txt>
    python echoroom.py server [--port 8000]
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

console = Console()
logging.basicConfig(level=logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────────────────────

async def cmd_join(args):
    """Join a meeting, transcribe live, print insights."""
    from bot.meeting_bot import BotConfig, MeetingBot
    from transcription.audio_capture import AudioCapture, AudioConfig
    from transcription.transcriber import TranscriptionConfig, WhisperTranscriber
    from intelligence.analyser import LiveAnalyser, ReportGenerator

    console.print(f"\n[bold cyan]🎙  EchoRoom joining:[/bold cyan] {args.url}\n")

    analyser = LiveAnalyser()
    transcript_lines = []

    def on_segment(seg):
        line = f"[bold]{seg.speaker}[/bold]: {seg.text}"
        console.print(line)
        transcript_lines.append(f"{seg.speaker}: {seg.text}")
        analyser.ingest(seg.speaker, seg.text, seg.start)

    transcriber = WhisperTranscriber(
        on_segment=on_segment,
        config=TranscriptionConfig(whisper_model=args.model),
    )
    transcriber.load()
    transcriber.start(time.time())

    audio = AudioCapture(
        on_chunk=transcriber.submit_chunk,
        config=AudioConfig(),
    )

    bot_cfg = BotConfig(bot_name=args.name, headless=not args.visible)
    bot = MeetingBot(bot_cfg)

    try:
        await bot.join(args.url)
        audio.start()
        console.print("[green]✅  Bot joined and recording. Press Ctrl+C to stop.[/green]\n")

        # Stream captions
        while True:
            try:
                entry = await asyncio.wait_for(bot.caption_queue.get(), 1.0)
                if entry.get("text"):
                    console.print(
                        f"[dim][caption][/dim] [bold]{entry['speaker']}[/bold]: {entry['text']}"
                    )
                    transcript_lines.append(f"{entry['speaker']}: {entry['text']}")
            except asyncio.TimeoutError:
                pass

    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping...[/yellow]")
    finally:
        audio.stop()
        transcriber.stop()
        await bot.leave()

        insights = analyser.finalise()
        _print_insights(insights)

        if args.save:
            _save_transcript(transcript_lines, args.save)
            _save_report(insights, transcript_lines, args.save)


def cmd_transcribe(args):
    """Transcribe a local WAV file."""
    from transcription.transcriber import TranscriptionConfig, WhisperTranscriber

    console.print(f"\n[cyan]Transcribing:[/cyan] {args.file}")
    segments = []

    def collect(seg):
        segments.append(seg)

    t = WhisperTranscriber(on_segment=collect,
                           config=TranscriptionConfig(whisper_model=args.model))
    t.load()
    t.transcribe_file(args.file)

    for seg in segments:
        console.print(f"[{seg.start:.1f}s] [bold]{seg.speaker}[/bold]: {seg.text}")

    if args.save:
        lines = [f"{s.speaker}: {s.text}" for s in segments]
        Path(args.save).write_text("\n".join(lines))
        console.print(f"[green]Saved to {args.save}[/green]")


def cmd_qa(args):
    """Interactive Q&A over a transcript file."""
    from intelligence.analyser import MeetingQA

    transcript = Path(args.file).read_text()
    qa = MeetingQA()
    qa.index(transcript.splitlines())
    qa.interactive_loop()


def cmd_server(args):
    """Start the FastAPI server."""
    import uvicorn
    console.print(f"[cyan]Starting EchoRoom API on port {args.port}...[/cyan]")
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=args.port,
        reload=args.dev,
        log_level="info",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_insights(insights):
    console.rule("[bold green]Meeting Insights")
    console.print(Panel(insights.summary or "No summary yet.", title="Summary"))

    if insights.action_items:
        t = Table("Owner", "Task", "Priority", "Due", title="Action Items")
        for ai in insights.action_items:
            t.add_row(ai.owner, ai.task, ai.priority, ai.due or "—")
        console.print(t)

    if insights.key_decisions:
        console.print("\n[bold]Key Decisions:[/bold]")
        for d in insights.key_decisions:
            console.print(f"  • {d}")

    if insights.speaker_stats:
        console.print("\n[bold]Speaking time (words):[/bold]")
        for speaker, words in sorted(
            insights.speaker_stats.items(), key=lambda x: -x[1]
        ):
            console.print(f"  {speaker}: {words} words")


def _save_transcript(lines, prefix):
    path = f"{prefix}_transcript.txt"
    Path(path).write_text("\n".join(lines))
    console.print(f"[green]Transcript saved to {path}[/green]")


def _save_report(insights, lines, prefix):
    from intelligence.analyser import ReportGenerator
    gen = ReportGenerator()
    report = gen.generate(insights, "\n".join(lines))
    path = f"{prefix}_report.md"
    Path(path).write_text(report)
    console.print(f"[green]Report saved to {path}[/green]")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(prog="echoroom", description="EchoRoom CLI")
    sub = parser.add_subparsers(dest="command")

    # join
    p_join = sub.add_parser("join", help="Join a meeting")
    p_join.add_argument("url", help="Meeting URL (Google Meet / Zoom / Teams)")
    p_join.add_argument("--name", default="EchoRoom Bot", help="Bot display name")
    p_join.add_argument("--model", default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size")
    p_join.add_argument("--visible", action="store_true",
                        help="Run browser in visible (non-headless) mode")
    p_join.add_argument("--save", metavar="PREFIX",
                        help="Save transcript + report with this filename prefix")

    # transcribe
    p_tr = sub.add_parser("transcribe", help="Transcribe a local audio file")
    p_tr.add_argument("file", help="Path to WAV file")
    p_tr.add_argument("--model", default="base")
    p_tr.add_argument("--save", metavar="OUTPUT", help="Save transcript to file")

    # qa
    p_qa = sub.add_parser("qa", help="Q&A over a transcript file")
    p_qa.add_argument("file", help="Path to transcript .txt file")

    # server
    p_srv = sub.add_parser("server", help="Start the API server")
    p_srv.add_argument("--port", type=int, default=8000)
    p_srv.add_argument("--dev", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "join":
        asyncio.run(cmd_join(args))
    elif args.command == "transcribe":
        cmd_transcribe(args)
    elif args.command == "qa":
        cmd_qa(args)
    elif args.command == "server":
        cmd_server(args)


if __name__ == "__main__":
    main()
