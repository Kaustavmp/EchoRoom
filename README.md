🎤 Echoroom – AI Meeting Assistant

Echoroom is an intelligent, web-based assistant that transcribes meeting audio and allows users to ask questions about what was discussed — even long after the meeting has ended.

🧩 Problem

Meetings often contain valuable insights, decisions, and action items, but most of that information gets lost once the call ends. Manually reviewing recordings or notes is time-consuming, inefficient, and error-prone.

✅ Solution
Echoroom automates the process of:
- Transcribing meeting audio using Whisper (via Hugging Face Transformers)
- Storing transcripts securely in MongoDB Atlas
- Enabling natural language questions over transcripts using OpenAI + LangChain

This allows users to **query meeting content like a chat with their own AI assistant**, saving time and improving team knowledge retention.

🚀 Features

- 🎙️ Upload `.wav` recordings of any meeting
- 🧠 Transcribe audio with Whisper (`openai/whisper-base`)
- 🗂️ Store and manage transcripts in MongoDB Atlas
- 🤖 Ask follow-up questions using LangChain + OpenAI
- 🖥️ Streamlit-powered interface — no coding required
