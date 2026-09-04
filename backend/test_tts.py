import torch
import soundfile as sf

from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer


MODEL_NAME = "ai4bharat/indic-parler-tts-pretrained"

print("Loading TTS model...")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained(
    MODEL_NAME
).to(device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"Model loaded successfully on {device}!")

# Text we want the AI to speak
text = "Hello, my name is Gurinder Singh. I am testing the Vaani Link application."

# Voice description
description = (
    "A clear Indian male voice speaking naturally and clearly."
)

input_ids = tokenizer(
    description,
    return_tensors="pt"
).input_ids.to(device)

prompt_input_ids = tokenizer(
    text,
    return_tensors="pt"
).input_ids.to(device)

print("Generating speech...")

with torch.no_grad():
    generation = model.generate(
        input_ids=input_ids,
        prompt_input_ids=prompt_input_ids
    )

audio = generation.cpu().numpy().squeeze()

OUTPUT_FILE = "english_output.wav"

sf.write(
    OUTPUT_FILE,
    audio,
    model.config.sampling_rate
)

print(f"Speech generated successfully: {OUTPUT_FILE}")