from pyannote.audio import Pipeline
import torch
from dotenv import load_dotenv
import os
from pyannote.audio.pipelines.utils.hook import ProgressHook
import torchaudio
import noisereduce as nr

load_dotenv()

token = os.getenv("HF_TOKEN")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=token
)
pipeline.to(torch.device("cuda"))

# Load and denoise audio
waveform, sample_rate = torchaudio.load("aepyx.wav")
audio_np = waveform.numpy()[0]
reduced = nr.reduce_noise(y=audio_np, sr=sample_rate)
clean_waveform = torch.tensor(reduced).unsqueeze(0)

# Apply pipeline with denoised audio
with ProgressHook() as hook:
    output = pipeline({"waveform": clean_waveform, "sample_rate": sample_rate}, hook=hook)

with open("output.rttm", "w") as f:
    output.speaker_diarization.write_rttm(f)
