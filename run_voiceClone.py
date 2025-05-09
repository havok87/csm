import os
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from generator import load_csm_1b, Segment
from dataclasses import dataclass

# Disable Triton compilation
os.environ["NO_TORCH_COMPILE"] = "1"

speakers = [0]
transcripts = [
    "This is test audio clip #1 transcript", 
    "This is a test audio clip #2 transcript",
]

audio_paths = [
	"/path/to/audioClip1.wav",
	"/path/to/audioClip2.wav",
]

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

generator = load_csm_1b(device=device)

def load_audio(audio_path):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    return audio_tensor

segments = [
    Segment(text=transcript, speaker=speaker, audio=load_audio(audio_path))
    for transcript, speaker, audio_path in zip(transcripts, speakers, audio_paths)
]
audio = generator.generate(
    text="This is what the generated text will say.",
    speaker=0,
    context=segments,
    max_audio_length_ms=90_000,
    temperature=0.65
)

torchaudio.save("csm_generated_audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate) #make sure to change the filename and path