import platform
import subprocess
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def get_meeting_url():
    url = input("Enter the meeting URL (Zoom, Google Meet, etc.): ").strip()
    return url

def join_meeting(meeting_url):
    chrome_options = Options()
    chrome_options.add_argument("--use-fake-ui-for-media-stream")  # Auto-enable mic/cam
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-popup-blocking")

    driver = webdriver.Chrome(options=chrome_options)
    driver.get(meeting_url)

    print("Joined the meeting... waiting for 60 seconds to ensure it starts")
    time.sleep(60)  # adjust if needed
    return driver

def detect_audio_device():
    system = platform.system()
    if system == "Darwin":  # macOS
        return ["-f", "avfoundation", "-i", ":0"]
    elif system == "Windows":
        return ["-f", "dshow", "-i", "audio='Stereo Mix'"]  # Adjust device name
    elif system == "Linux":
        return ["-f", "alsa", "-i", "default"]
    else:
        raise RuntimeError("Unsupported OS")

def record_audio(filename="meeting_audio.wav", duration=3600):
    print("Recording meeting audio...")
    input_device = detect_audio_device()
    command = ["ffmpeg"] + input_device + ["-t", str(duration), filename]
    subprocess.run(command)
    print("Audio recording complete.")

if __name__ == "__main__":
    url = get_meeting_url()
    driver = join_meeting(url)
    record_audio()
    driver.quit()
