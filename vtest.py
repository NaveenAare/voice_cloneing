import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize TTS with the specified model
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Print the model name
print(f"Using model: {tts.model_name}")

# Print the model's configuration (if needed)
print(f"Model configuration: {tts.config}")

# Get the model file paths
model_files = tts.get_models_file_path()
print("Model files:")
for file in model_files:
    print(file)

# Run TTS
wav = tts.tts(text="Hello world!", speaker_wav="my/cloning/audio.wav", language="en")
tts.tts_to_file(text="Hello world!", speaker_wav="my/cloning/audio.wav", language="en", file_path="output.wav")
