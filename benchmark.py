"""
benchmark.py
Naloži pretrained pyannote model in izračuna DER na AMI test setu.
Primerjava z rezultati iz papirja (Bredin et al. 2019): DER = 29.7%

Namestitev:
    pip install pyannote.audio pyannote.metrics torch torchaudio python-dotenv

HuggingFace token:
    1. Ustvarite račun na huggingface.co
    2. Sprejmite pogoje: huggingface.co/pyannote/speaker-diarization-3.1
    3. Sprejmite pogoje: huggingface.co/pyannote/segmentation-3.0
    4. Token: huggingface.co/settings/tokens
    5. Dodajte v .env: HF_TOKEN=hf_xxxxxxxx
"""

import os
import torch
import torchaudio
from pathlib import Path
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate

load_dotenv()

# ── NASTAVITVE ────────────────────────────────
HF_TOKEN  = os.getenv("HF_TOKEN")
AUDIO_DIR = Path("data/ami/test/audio")
RTTM_DIR  = Path("data/ami/test/rttm")
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ─────────────────────────────────────────────

# 1. Naloži model
print(f"Nalagam model na {DEVICE}...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=HF_TOKEN          # 'token' namesto starega 'use_auth_token'
)
pipeline = pipeline.to(DEVICE)

# 2. Poženi na vseh test filih
metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)

for audio_path in sorted(AUDIO_DIR.glob("*.wav")):
    uri = audio_path.stem
    rttm_path = RTTM_DIR / f"{uri}.rttm"

    if not rttm_path.exists():
        print(f"Preskočim {uri} — ni RTTM")
        continue

    print(f"Obdelujem {uri}...")

    # Preloadi audio z torchaudio — izogneš se torchcodec problemu
    waveform, sample_rate = torchaudio.load(str(audio_path))
    audio_input = {"waveform": waveform, "sample_rate": sample_rate}

    # Referenca (ground truth)
    reference = list(load_rttm(str(rttm_path)).values())[0]

    # Hipoteza (naš model) — podamo preloadan audio namesto poti
    hypothesis = pipeline(audio_input)

    # DER za ta file
    der = metric(reference, hypothesis, detailed=True)
    print(f"  {uri}: DER = {der['diarization error rate']*100:.1f}%")

# 3. Skupni rezultat
overall = abs(metric) * 100
print(f"\nSKUPNI DER : {overall:.1f}%")
print(f"PAPIR DER  : 29.7%")
print(f"RAZLIKA    : {overall - 29.7:+.1f}%")