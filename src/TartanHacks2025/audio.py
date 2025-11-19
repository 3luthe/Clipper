from IPython.display import display, Image, Audio

import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import time
import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY is not set. Please configure it in your environment before running this script.")

def reader(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

script = "script.txt"

result = reader(script)

response = requests.post(
    "https://api.openai.com/v1/audio/speech",
    headers={
        "Authorization": f"Bearer {api_key}",
    },
    json={
        "model": "tts-1-1106",
        "input": result,
        "voice": "onyx",
    },
)

audio = b""
for chunk in response.iter_content(chunk_size=1024 * 1024):
    audio += chunk

audio_filename = "audio/output_audio.mp3"

# Save the audio data to a file
with open(audio_filename, "wb") as audio_file:
    audio_file.write(audio)

print(f"Audio saved as {audio_filename}")