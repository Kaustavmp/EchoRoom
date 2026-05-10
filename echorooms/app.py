import os
import streamlit as st
from database import get_transcripts, save_transcript
from transformers import pipeline
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# Load OpenAI API key from a custom file path
import sys

sys.path.append(r"C:\Users\KAUSTAV\OneDrive\Desktop\EchoRoom")
from Api_Key import api_key as OPENAI_API_KEY

st.set_page_config(page_title="Echoroom Assistant", layout="centered")
st.title("🎤 Echoroom AI Meeting Assistant")

uploaded_audio = st.file_uploader("Upload Meeting Audio File (.wav)", type=["wav"])

if uploaded_audio:
    st.success("Audio uploaded. Beginning transcription...")

    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_audio.read())

    # Use HuggingFace Transformers pipeline for transcription as a replacement for Whisper
    transcriber = pipeline("automatic-speech-recognition")
    with open("temp_audio.wav", "rb") as audio_file:
        transcript = transcriber(audio_file)["text"]

    st.text_area("📝 Transcript Preview", transcript[:2000] + "...", height=250)

    if st.button("💾 Save Transcript to MongoDB"):
        save_transcript("Uploaded via Web", transcript)
        st.success("Saved successfully to MongoDB!")

    st.divider()

    st.header("🔎 Ask Questions About This Meeting")
    q = st.text_input("Your question:")
    if q:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = splitter.split_text(transcript)
        docs = [Document(page_content=t) for t in texts]

        db = FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
        qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY), retriever=db.as_retriever())
        st.success(qa.run(q))

# ──────────────────────────────────────────────
# Past Transcripts Section
# ──────────────────────────────────────────────
st.divider()
past = get_transcripts()
if past:
    st.header("🗂️ Previous Meeting Transcripts")
    options = {f"{p['timestamp'].strftime('%Y-%m-%d %H:%M')} - {p.get('meeting_url', 'Manual Upload')}": p for p in
               past}
    selection = st.selectbox("Select a transcript to query:", list(options.keys()))
    selected_doc = options[selection]
    selected_transcript = selected_doc["transcript"]

    q2 = st.text_input("Ask a question about the selected meeting:")
    if q2:
        texts = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(selected_transcript)
        docs = [Document(page_content=t) for t in texts]
        db = FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
        qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY), retriever=db.as_retriever())
        st.success(qa.run(q2))
