"""
benchmark.py
Naloži pretrained pyannote model in izračuna DER na AMI test setu.

Namestitev:
    pip install pyannote.audio pyannote.metrics torch torchaudio

"""

from pathlib import Path
from pyannote.audio import Pipeline
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate
import torch

HF_TOKEN     = "VAŠ_TOKEN_TUKAJ"
AUDIO_DIR    = Path("data/ami/test/audio")
RTTM_DIR     = Path("data/ami/test/rttm")

# run model
print("Nalagam model...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN
)
pipeline = pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# run on test files
metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)

for audio_path in sorted(AUDIO_DIR.glob("*.wav")):
    uri = audio_path.stem.replace(".Mix-Headset", "")
    rttm_path = RTTM_DIR / f"{uri}.rttm"

    if not rttm_path.exists():
        print(f"Preskočim {uri} — ni RTTM")
        continue

    reference = list(load_rttm(str(rttm_path)).values())[0]

    hypothesis = pipeline(str(audio_path))

    der = metric(reference, hypothesis, detailed=True)
    print(f"{uri}: DER = {der['diarization error rate']*100:.1f}%")

overall = abs(metric) * 100
print(f"\nSKUPNI DER : {overall:.1f}%")
print(f"PAPIR DER  : 29.7%")
print(f"RAZLIKA    : {overall - 29.7:+.1f}%")