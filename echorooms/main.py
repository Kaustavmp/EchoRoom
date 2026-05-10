import os
import time
import subprocess
import platform
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

import whisper
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

from database import init_db, save_transcript
from models import OPENAI_KEY, AUDIO_FILE, TRANSCRIPT_MODEL, RECORD_DURATION


def get_meeting_url():
    return input("Enter the meeting URL: ").strip()

def join_meeting(url):
    chrome_options = Options()
    chrome_options.add_argument("--use-fake-ui-for-media-stream")
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    print("Joining meeting... waiting 60 sec")
    time.sleep(60)
    return driver

def detect_audio_device():
    system = platform.system()
    if system == "Darwin":
        return ["-f", "avfoundation", "-i", ":0"]
    elif system == "Windows":
        return ["-f", "dshow", "-i", "audio='Stereo Mix'"]
    elif system == "Linux":
        return ["-f", "alsa", "-i", "default"]
    raise RuntimeError("Unsupported OS")

def record_audio(filename=AUDIO_FILE, duration=RECORD_DURATION):
    print("Recording audio...")
    device_args = detect_audio_device()
    subprocess.run(["ffmpeg"] + device_args + ["-t", str(duration), filename])
    print("Recording finished.")

def transcribe_audio(path=AUDIO_FILE, model_name=TRANSCRIPT_MODEL):
    model = whisper.load_model(model_name)
    print("Transcribing...")
    return model.transcribe(path)["text"]

def qa_interface(transcript):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_text(transcript)
    docs = [Document(page_content=t) for t in texts]

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
    vectordb = FAISS.from_documents(docs, embeddings)
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(openai_api_key=OPENAI_KEY), retriever=vectordb.as_retriever())

    print("\nAsk anything about the meeting transcript (type 'exit' to quit):")
    while True:
        q = input("\nQ: ")
        if q.lower() == "exit":
            break
        print("A:", qa.run(q))

if __name__ == "__main__":
    init_db()
    url = get_meeting_url()
    driver = join_meeting(url)
    record_audio()
    driver.quit()

    transcript = transcribe_audio()
    save_transcript(url, transcript)

    print("\n--- Transcript preview ---\n")
    print(transcript[:1000] + "...\n")
    qa_interface(transcript)
