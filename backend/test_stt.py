from transformers import AutoModel
import torch
import soundfile as sf

MODEL_NAME = "ai4bharat/indic-conformer-600m-multilingual"
AUDIO_PATH = "recording.wav"
LANGUAGE = "hi"

print("Loading IndicConformer model...")

model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model.eval()

print("Model loaded successfully!")

print("\nLoading audio...")

audio, sample_rate = sf.read(AUDIO_PATH)

print(f"Sample rate: {sample_rate}")
print(f"Audio shape: {audio.shape}")

# Convert NumPy audio to PyTorch tensor
wav = torch.tensor(audio, dtype=torch.float32)

# Ensure mono audio
if wav.ndim > 1:
    wav = wav.mean(dim=1)

# Add batch dimension
wav = wav.unsqueeze(0)

print("\nTranscribing...")

with torch.no_grad():
    transcription = model(
        wav,
        LANGUAGE,
        "ctc"
    )

print("\n--- TRANSCRIPTION ---")
print(transcription)