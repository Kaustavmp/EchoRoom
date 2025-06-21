from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# MongoDB connection details
MONGO_URI = os.getenv("MONGODB_URI", "mongodb+srv://HeadsBosses:Kaustav@echodb.pr3ktbk.mongodb.net/")
DB_NAME = os.getenv("MONGO_DB_NAME", "echoroom")
COLLECTION = os.getenv("MONGO_COLLECTION", "transcripts")

# Connect to MongoDB Atlas
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION]

def save_transcript(meeting_url: str, transcript: str):

    doc = {
        "meeting_url": meeting_url,
        "timestamp": datetime.utcnow(),
        "transcript": transcript
    }
    collection.insert_one(doc)

def get_transcripts():
    return list(collection.find().sort("timestamp", -1))
