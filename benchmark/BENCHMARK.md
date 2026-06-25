# Artur-J Diarization & ASR Benchmark

## Goal

Compare two pipelines on Slovenian speech audio against gold-standard labels:

| | Our Pipeline | WhisperX |
|---|---|---|
| Diarization | pyannote/speaker-diarization-3.1 | pyannote via WhisperX |
| ASR | samolego/whisper-small-slovenian | whisper large-v3 |
| Preprocessing | preprocess.py (filters, normalization) | none |

Metrics: **DER** (speaker diarization error rate) and **WER** (word error rate).

---

## Dataset

`datasets/artur-j/` — Slovenian public speech (media, conferences, education).

```
datasets/artur-j/
├── wav/    ← audio files (Artur-J-Gvecg-PXXXXXX-avd.wav)
├── rttm/   ← gold diarization labels (.rttm, same base name as wav)
└── trs/    ← gold transcriptions (.trs, same base name but -std suffix)
```

---

## Running

```bash
# Single file
python benchmark/benchmark_single.py datasets/artur-j/wav/Artur-J-Gvecg-P500002-avd.wav

# Full loop — skips already-completed files, Ctrl+C exits immediately
python benchmark/benchmark_loop.py

# Random order
python benchmark/benchmark_loop.py --shuffle
```

Results are saved to `benchmark/results/<file_id>.json` after each file.

---

## Per-file Output

```
============================================================
File: Artur-J-Gvecg-P500002-avd
============================================================

[WhisperX] Running...
  Done in 142.3s (87 segments)
  DER (collar=0.0): 12.34%  Miss=8.10%  FA=1.20%  Conf=3.04%
  DER (collar=0.25): 9.87%  ...
  WER: 34.21%  (ref=732 words, hyp=698 words)

[OurPipeline] Running...
  Done in 61.1s (33 segments)
  [OurPipeline] preprocess=0.3s  diarization=42.1s  asr=18.7s  total=61.1s  (audio=423.0s  RTF=0.14x)
  DER (collar=0.0): 5.95%  Miss=5.95%  FA=0.00%  Conf=0.00%
  WER: 45.12%  (ref=732 words, hyp=575 words)

────────────────────────────────────────
SUMMARY
  DER:  WhisperX=12.34%  OurPipeline=5.95%  Δ=-6.39%  → OurPipeline wins
  WER:  WhisperX=34.21%  OurPipeline=45.12%  Δ=+10.91%  → WhisperX wins

  Saved: benchmark/results/Artur-J-Gvecg-P500002-avd.json
```

---

## Metrics Explained

**DER** — Diarization Error Rate. Breakdown:
- **Miss** — speech present but not detected
- **FA** — silence labelled as speech
- **Confusion** — right time, wrong speaker assigned

Computed at two collars:
- `collar=0.0` — strict, no boundary forgiveness
- `collar=0.25` — lenient, 250ms boundary tolerance (standard in literature)

**WER** — Word Error Rate. Both pipelines' text is concatenated in time order, lowercased, punctuation stripped, then compared against the gold TRS transcript with `jiwer`.

**RTF** — Real-Time Factor. `total_time / audio_duration`. Below 1.0 = faster than real time.

---

## File Structure

```
benchmark/
├── utils/
│   ├── run_our_pipeline.py   ← our pipeline (pyannote 3.1 + whisper-sl)
│   ├── run_whisperx.py       ← WhisperX (large-v3 + pyannote diarization)
│   ├── evaluate_der.py       ← DER computation
│   ├── evaluate_wer.py       ← WER computation
│   ├── rttm_io.py            ← load RTTM, convert segments to Annotation
│   └── trs_parser.py         ← parse TRS reference transcriptions
├── benchmark_single.py       ← benchmark one file
├── benchmark_loop.py         ← loop over all files
├── results/                  ← JSON result per file
└── BENCHMARK.md              ← this file
```

---

## Requirements

```bash
pip install -r benchmark/requirements.txt
```

HuggingFace token required for pyannote models:
```bash
# add to .env file in project root
HF_TOKEN=your_token_here
```

---

## Known Issues

- **torchaudio 2.11+** removed several APIs pyannote depends on — patched automatically in `benchmark_single.py` at startup.
- **PyTorch 2.6+** changed `torch.load` default to `weights_only=True` which breaks pyannote checkpoints — patched to force `weights_only=False`.
- **SpeechBrain** optional integrations (k2, nlp) crash on lazy import if not installed — stubbed out automatically.
- **WhisperX** requires cuDNN 8.x via ctranslate2. If `cudnn_ops_infer64_8.dll` is missing, upgrade ctranslate2: `pip install ctranslate2 --upgrade`.
