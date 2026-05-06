from pyannote.audio import Pipeline
import torch

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token="hf_..."
)
pipeline = pipeline.to(torch.device("cuda"))

diarization = pipeline("data/ami/test/audio/meeting_000.wav")

for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{speaker}: {turn.start:.1f}s – {turn.end:.1f}s")