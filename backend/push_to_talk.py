import sounddevice as sd
import soundfile as sf

from stt_service import transcribe_audio

SAMPLE_RATE = 16000
AUDIO_FILE = "push_to_talk.wav"

print("Press ENTER to START recording...")
input()

print("🎤 Recording... Speak now!")
print("Press ENTER again when finished.")

# Start recording continuously
audio_chunks = []

def callback(indata, frames, time, status):
    if status:
        print(status)

    audio_chunks.append(indata.copy())


# Start microphone stream
with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    callback=callback
):

    input()


print("Recording stopped!")

# Combine all recorded chunks
import numpy as np

audio = np.concatenate(audio_chunks, axis=0)

# Save audio
sf.write(AUDIO_FILE, audio, SAMPLE_RATE)

print("Audio saved!")

print("\n🧠 Transcribing...")

text = transcribe_audio(AUDIO_FILE, "hi")

print("\n--- TRANSCRIPTION ---")
print(text)