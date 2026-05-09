from pyannote.audio import Pipeline
import torch
from dotenv import load_dotenv
import os
from pyannote.audio.pipelines.utils.hook import ProgressHook

load_dotenv()

token = os.getenv("HF_TOKEN")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=token
)

# send pipeline to GPU (when available)
pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) # type: ignore

# apply pretrained pipeline (with optional progress hook)
with ProgressHook() as hook:
    output = pipeline("aepyx.wav", hook=hook) # type: ignore

# with open("diarization.txt", "w") as f:
#   for turn, speaker in output.speaker_diarization:
#        f.write(f"{speaker} speaks between t={turn.start:.3f}s and t={turn.end:.3f}s\n")

with open("output.rttm", "w") as f:
    output.speaker_diarization.write_rttm(f)
