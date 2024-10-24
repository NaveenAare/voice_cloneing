from flask import Flask, request, send_file, jsonify
import os
import torch
from TTS.api import TTS
import requests
import tempfile

app = Flask(__name__)

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize TTS model
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Directory to store downloaded audio files permanently
PERMANENT_AUDIO_DIR = "downloaded_audios"
os.makedirs(PERMANENT_AUDIO_DIR, exist_ok=True)

def download_audio(audio_name, audio_url):

    audio_path = os.path.join(PERMANENT_AUDIO_DIR, audio_name)

    if not os.path.exists(audio_path):


        response = requests.get(audio_url)
        print("in downloading........................................................................................................")
        if response.status_code == 200:
            with open(audio_path, 'wb') as f:
                f.write(response.content)
            return audio_path
        else:
            return None
    return audio_path  # Return the existing file path

@app.route('/tts', methods=['POST'])
def text_to_speech():
    data = request.json
    text = data.get('text')
    audio_name = data.get('audio_name')
    audio_url = data.get('audio_url')

    if not text or not audio_name or not audio_url:
        return jsonify({"error": "Text, audio_name, and audio_url are required."}), 400

    audio_path = download_audio(audio_name, audio_url)
    if not audio_path:
        return jsonify({"error": "Failed to download audio from the provided URL."}), 400

    # Run TTS and return the output as a response
    try:
        output_file = f"output_{os.urandom(4).hex()}.wav"
        output_path = os.path.join(tempfile.gettempdir(), output_file)

        # Generate audio with TTS
        wav = tts.tts(text=text, speaker_wav=audio_path, language="en")

        # Save the audio to a file
        tts.tts_to_file(text=text, speaker_wav=audio_path, language="en", file_path=output_path)

        # Send the generated audio file as a response
        return send_file(output_path, as_attachment=True, download_name=output_file)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
