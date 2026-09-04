from transformers import AutoModel
import torch
import soundfile as sf

MODEL_NAME = "ai4bharat/indic-conformer-600m-multilingual"

print("Loading IndicConformer model...")

model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model.eval()

print("STT model loaded successfully!")


def transcribe_audio(audio_path, language="hi"):
    """
    Convert a WAV audio file into text.
    """

    audio, sample_rate = sf.read(audio_path)

    # Convert to PyTorch tensor
    wav = torch.tensor(audio, dtype=torch.float32)

    # Convert stereo to mono if needed
    if wav.ndim > 1:
        wav = wav.mean(dim=1)

    # Add batch dimension
    wav = wav.unsqueeze(0)

    # Run Speech-to-Text
    with torch.no_grad():
        transcription = model(
            wav,
            language,
            "ctc"
        )

    return transcription