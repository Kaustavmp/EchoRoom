from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
AUDIO_FILE = "meeting_audio.wav"
TRANSCRIPT_MODEL = "base"
RECORD_DURATION = 3600  # seconds
