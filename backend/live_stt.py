import sounddevice as sd
import soundfile as sf

from stt_service import transcribe_audio

SAMPLE_RATE = 16000
DURATION = 5
AUDIO_FILE = "live_recording.wav"

print("🎤 Recording will start now...")
print(f"Speak for {DURATION} seconds!")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32"
)

sd.wait()

print("Recording finished!")

# Save audio
sf.write(AUDIO_FILE, audio, SAMPLE_RATE)

print("\nTranscribing...")

text = transcribe_audio(AUDIO_FILE, "hi")

print("\n--- TRANSCRIPTION ---")
print(text)