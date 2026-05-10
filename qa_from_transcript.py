import os
import sys
import importlib.util

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from typing import List

def load_api_key_from_file():
    api_key_path = r"C:\Users\KAUSTAV\OneDrive\Desktop\EchoRoom\Api_Key.py"
    spec = importlib.util.spec_from_file_location("api_key_module", api_key_path)
    api_key_module = importlib.util.module_from_spec(spec)
    sys.modules["api_key_module"] = api_key_module
    spec.loader.exec_module(api_key_module)

    os.environ["OPENAI_API_KEY"] = api_key_module.api_key
    print("🔑 OpenAI API Key loaded successfully.")


def transcribe_audio():
    print("🎙️ Starting transcription...")
    return "This is a placeholder transcript. Replace with actual audio transcription logic."


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()


def qa_from_transcript(transcript_text):
    if not transcript_text.strip():
        print("❌ Transcript is empty. Exiting.")
        return

    # Step 1: Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(transcript_text)
    documents = [Document(page_content=chunk) for chunk in chunks]

    # Step 2: Create FAISS Vector Store
    embedding_model = SentenceTransformerEmbeddings()
    try:
        vectorstore = FAISS.from_documents(documents, embedding=embedding_model)
    except Exception as e:
        print("❌ VectorStore error:", e)
        return

    # Step 3: Setup QA chain
    try:
        retriever = vectorstore.as_retriever()
        llm = ChatOpenAI(temperature=0)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    except Exception as e:
        print("❌ QA Chain error:", e)
        return

    # Step 4: Query loop
    print("\n🤖 EchoRoom QA is ready. Ask any question or type 'exit' to quit.")
    while True:
        try:
            query = input("\n❓ Ask: ")
            if query.strip().lower() == "exit":
                print("👋 Exiting EchoRoom QA.")
                break
            response = qa_chain.run(query)
            if not response or not response.strip().lower() in ["i do not know", "i don't know",""]:
                print("I am sorry, I don't know the answer to that.")
            else:
                print("✅ Answer:", response)
        except Exception as e:
            print("⚠️ Error during Q&A:", e)


if __name__ == "__main__":
    load_api_key_from_file()

    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError("❌ OPENAI_API_KEY is not set in the environment.")

    transcript = transcribe_audio()
    qa_from_transcript(transcript)
