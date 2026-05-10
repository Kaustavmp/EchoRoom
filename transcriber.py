import whisper

def transcribe_audio(audio_path="meeting_audio.wav"):
    model = whisper.load_model("base")  # You can use "small", "medium", or "large"
    print("Transcribing audio...")
    result = model.transcribe(audio_path)
    return result["text"]

if __name__ == "__main__":
    transcript = transcribe_audio()
    print("\nTranscription:\n")
    print(transcript)
