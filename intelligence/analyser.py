"""
EchoRoom – Intelligence Layer
Real-time and post-meeting analysis:
  • Live summary (rolling window)
  • Action item extraction
  • Sentiment analysis & flagging
  • Speaker statistics
  • RAG-powered Q&A over the full transcript
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

from openai import OpenAI

logger = logging.getLogger("echoroom.intelligence")


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ActionItem:
    owner: str
    task: str
    due: Optional[str] = None
    priority: str = "medium"   # low / medium / high


@dataclass
class SentimentFlag:
    timestamp: float
    speaker: str
    text: str
    sentiment: str   # positive / negative / frustrated / confused / excited
    score: float     # –1.0 to 1.0


@dataclass
class MeetingInsights:
    summary: str = ""
    key_decisions: List[str] = field(default_factory=list)
    action_items: List[ActionItem] = field(default_factory=list)
    sentiment_flags: List[SentimentFlag] = field(default_factory=list)
    speaker_stats: dict = field(default_factory=dict)   # speaker → word count
    topics: List[str] = field(default_factory=list)
    follow_up_questions: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Live analysis (rolling window)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are EchoRoom, an intelligent meeting assistant.
Analyse the provided meeting transcript segment and return a JSON object with:
{
  "summary": "2-3 sentence running summary",
  "key_decisions": ["decision 1", ...],
  "action_items": [
    {"owner": "name or Unknown", "task": "task description", "due": "date or null", "priority": "low|medium|high"}
  ],
  "sentiment_flags": [
    {"speaker": "name", "sentiment": "positive|negative|frustrated|confused|excited", "score": 0.0, "text": "excerpt"}
  ],
  "topics": ["topic 1", "topic 2"]
}
Only include non-empty arrays. Be concise and precise. Return ONLY valid JSON."""


class LiveAnalyser:
    """
    Buffers incoming TranscriptSegments and periodically sends a window
    to the LLM for incremental insights.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        window_words: int = 600,
        overlap_words: int = 100,
    ):
        self._client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self._model = model
        self._window_words = window_words
        self._overlap_words = overlap_words
        self._buffer: List[str] = []      # buffered "Speaker: text" lines
        self._word_count = 0
        self.insights = MeetingInsights()

    def ingest(self, speaker: str, text: str, timestamp: float) -> None:
        """Call for each new TranscriptSegment."""
        line = f"{speaker}: {text}"
        self._buffer.append(line)
        self._word_count += len(text.split())

        # Update speaker word stats
        stats = self.insights.speaker_stats
        stats[speaker] = stats.get(speaker, 0) + len(text.split())

        if self._word_count >= self._window_words:
            self._analyse_window()

    def finalise(self) -> MeetingInsights:
        """Run one last analysis pass over any remaining buffer."""
        if self._buffer:
            self._analyse_window(force=True)
        return self.insights

    def _analyse_window(self, force: bool = False) -> None:
        window = "\n".join(self._buffer)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": window},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            raw = response.choices[0].message.content
            data = json.loads(raw)
            self._merge(data)
            logger.debug("Analysis window processed. Topics: %s",
                         data.get("topics", []))
        except Exception as exc:
            logger.error("LLM analysis failed: %s", exc)

        # Keep overlap for context continuity
        overlap_lines = self._trim_to_words(self._buffer, self._overlap_words)
        self._buffer = overlap_lines
        self._word_count = sum(
            len(line.split(": ", 1)[-1].split()) for line in overlap_lines
        )

    def _merge(self, data: dict) -> None:
        ins = self.insights
        if data.get("summary"):
            ins.summary = data["summary"]   # always take the latest summary
        for d in data.get("key_decisions", []):
            if d not in ins.key_decisions:
                ins.key_decisions.append(d)
        for ai in data.get("action_items", []):
            ins.action_items.append(ActionItem(**ai))
        for sf in data.get("sentiment_flags", []):
            ins.sentiment_flags.append(SentimentFlag(
                timestamp=time.time(),
                speaker=sf.get("speaker", "Unknown"),
                text=sf.get("text", ""),
                sentiment=sf.get("sentiment", "neutral"),
                score=sf.get("score", 0.0),
            ))
        for t in data.get("topics", []):
            if t not in ins.topics:
                ins.topics.append(t)

    @staticmethod
    def _trim_to_words(lines: List[str], n_words: int) -> List[str]:
        """Return the last n_words worth of lines."""
        total, out = 0, []
        for line in reversed(lines):
            words = len(line.split())
            if total + words > n_words:
                break
            out.insert(0, line)
            total += words
        return out


# ─────────────────────────────────────────────────────────────────────────────
# RAG-powered Q&A (post-meeting)
# ─────────────────────────────────────────────────────────────────────────────

QA_SYSTEM_PROMPT = """You are EchoRoom, a meeting assistant.
Answer questions about the meeting using ONLY the provided context.
If the answer is not in the context, say "I couldn't find that in the transcript."
Be concise and cite the speaker when relevant."""


class MeetingQA:
    """
    Retrieval-Augmented Q&A over the full meeting transcript.
    Uses FAISS + sentence-transformers for embedding, GPT-4 for answering.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self._client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self._model = model
        self._vectorstore = None
        self._chunks: List[str] = []

    def index(self, transcript_lines: List[str]) -> None:
        """Build the FAISS index from speaker-tagged transcript lines."""
        from langchain_community.vectorstores import FAISS          # type: ignore
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
        from langchain_core.documents import Document               # type: ignore
        from langchain.embeddings.base import Embeddings            # type: ignore
        from sentence_transformers import SentenceTransformer       # type: ignore

        class _STEmbeddings(Embeddings):
            def __init__(self):
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
            def embed_documents(self, texts):
                return self._model.encode(texts).tolist()
            def embed_query(self, text):
                return self._model.encode([text])[0].tolist()

        full_text = "\n".join(transcript_lines)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=60
        )
        chunks = splitter.split_text(full_text)
        docs = [Document(page_content=c) for c in chunks]
        embeddings = _STEmbeddings()
        self._vectorstore = FAISS.from_documents(docs, embeddings)
        logger.info("QA index built with %d chunks.", len(chunks))

    def ask(self, question: str, k: int = 5) -> str:
        """Answer a natural-language question about the meeting."""
        if self._vectorstore is None:
            return "Please call .index() first with the meeting transcript."

        docs = self._vectorstore.similarity_search(question, k=k)
        context = "\n\n".join(d.page_content for d in docs)

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": QA_SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()

    def interactive_loop(self) -> None:
        """CLI Q&A loop – runs until user types 'exit'."""
        print("\n🤖  EchoRoom Q&A ready. Ask anything about the meeting.")
        while True:
            try:
                q = input("\n❓  You: ").strip()
                if q.lower() in ("exit", "quit", "q"):
                    print("👋  Goodbye!")
                    break
                if not q:
                    continue
                answer = self.ask(q)
                print(f"✅  EchoRoom: {answer}")
            except (KeyboardInterrupt, EOFError):
                break


# ─────────────────────────────────────────────────────────────────────────────
# Post-meeting report generator
# ─────────────────────────────────────────────────────────────────────────────

REPORT_PROMPT = """You are EchoRoom. Write a professional, structured meeting report
in Markdown based on the insights and full transcript provided.
Include sections: Executive Summary, Key Decisions, Action Items (table),
Discussion Highlights (with speaker names), Sentiment Notes, Next Steps.
Be thorough but avoid unnecessary padding."""


class ReportGenerator:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self._client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self._model = model

    def generate(self, insights: MeetingInsights, transcript: str) -> str:
        payload = {
            "insights": {
                "summary": insights.summary,
                "key_decisions": insights.key_decisions,
                "action_items": [
                    {"owner": a.owner, "task": a.task,
                     "due": a.due, "priority": a.priority}
                    for a in insights.action_items
                ],
                "topics": insights.topics,
                "speaker_stats": insights.speaker_stats,
            },
            "transcript_excerpt": transcript[:8000],  # respect context window
        }

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": REPORT_PROMPT},
                {"role": "user", "content": json.dumps(payload)},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content
