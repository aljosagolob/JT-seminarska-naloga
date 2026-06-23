# AI Instructions: Artur-J Diarization & ASR Benchmark

## What We Have

```
data/datasets/artur-j/
├── wav/       ← audio files (.wav)
├── rttm/      ← gold diarization labels (.rttm, one per wav)
└── trs/       ← transcription files (.trs, one per wav)
```

Artur-J is the **public speech** subset of ARTUR 1.0 (Slovenian): media recordings,
conference talks, workshops, education videos. Multiple speakers per file.

## What We Are Building

A benchmark comparing two pipelines on Artur-J audio against gold labels:

- **Diarization quality** — DER (against gold rttm)
- **ASR quality** — WER (against trs transcriptions)

Systems under test:
- **WhisperX** — baseline (faster-whisper ASR + pyannote diarization)
- **Our pipeline** — custom system, same output format

**We do NOT run the full dataset at once.** Instead we build two scripts:
- `benchmark_single.py` — benchmark one specific WAV file, print results, exit
- `benchmark_loop.py` — pick files one by one and benchmark until Ctrl+C is pressed

---

## Project Structure to Create

```
benchmark/
├── utils/
│   ├── __init__.py
│   ├── trs_parser.py          ← extract reference text from TRS
│   ├── rttm_io.py             ← load/write RTTM files
│   ├── run_whisperx.py        ← run WhisperX on one audio file
│   ├── run_our_pipeline.py    ← run our pipeline on one audio file
│   ├── evaluate_der.py        ← compute DER for one file
│   └── evaluate_wer.py        ← compute WER for one file
├── benchmark_single.py        ← CLI: benchmark one WAV file
├── benchmark_loop.py          ← runs continuously until Ctrl+C
├── results/                   ← all results saved here as JSON
└── requirements.txt
```

---

## Requirements

```
# requirements.txt
pyannote.audio>=3.1.0
pyannote.metrics>=3.2.0
faster-whisper>=1.0.0
whisperx>=3.1.0
jiwer>=3.0.0
lxml>=4.9.0
pandas>=2.0.0
tqdm>=4.65.0
torch>=2.0.0
```

Install: `pip install -r requirements.txt`

WhisperX requires a HuggingFace token. Set: `export HF_TOKEN=your_token`

GPU is used automatically when available. All scripts detect CUDA and fall back to CPU.

---

## Shared Output Format

Both pipelines return a list of dicts. This is the internal format used everywhere:

```python
[
    {
        "speaker": "SPEAKER_0",   # arbitrary label, need not match gold
        "start":   1.240,         # seconds, float
        "end":     4.870,         # seconds, float
        "text":    "Dobrodošli na konferenci."
    },
    ...
]
```

Segments must be sorted by `start`. Speaker labels do not need to match between
systems or match the gold RTTM — DER evaluation resolves the optimal mapping automatically.

---

## `utils/__init__.py`

Empty file:
```python
```

---

## `utils/rttm_io.py`

Load gold RTTM files and convert pipeline output to RTTM format.

```python
from pyannote.core import Annotation, Segment
import os


def load_rttm(rttm_path: str) -> Annotation:
    """Load an RTTM file into a pyannote Annotation."""
    ann = Annotation()
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 10 or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            ann[Segment(start, start + duration)] = speaker
    return ann


def segments_to_annotation(segments: list) -> Annotation:
    """Convert pipeline output (list of dicts) to a pyannote Annotation."""
    ann = Annotation()
    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        if end - start <= 0.01:
            continue
        ann[Segment(start, end)] = seg["speaker"]
    return ann


def find_rttm(file_id: str, rttm_dir: str) -> str | None:
    """Find the gold RTTM path for a given file_id. Returns None if not found."""
    path = os.path.join(rttm_dir, f"{file_id}.rttm")
    return path if os.path.exists(path) else None
```

---

## `utils/trs_parser.py`

Extract reference text segments from a TRS file.

```python
from lxml import etree


def parse_trs(trs_path: str) -> list:
    """
    Parse a TRS file and return a list of reference segments:
    [{"speaker": str, "start": float, "end": float, "text": str}, ...]

    NOTE: TRS structure can vary. If text extraction looks wrong on your files,
    share a sample TRS snippet and ask for a fix to this parser.
    """
    try:
        tree = etree.parse(trs_path)
        root = tree.getroot()
    except Exception as e:
        raise ValueError(f"Failed to parse TRS file {trs_path}: {e}")

    segments = []

    for turn in root.findall(".//Turn"):
        start = float(turn.get("startTime", 0))
        end = float(turn.get("endTime", 0))
        speaker = turn.get("speaker", "UNKNOWN")

        # Collect text: Turn's own text node + all descendant text/tail nodes
        # Skip structural tags (Sync, Event, Who) — they carry no transcript text
        SKIP_TAGS = {"Sync", "Event", "Who", "Turn", "Section", "Episode"}
        text_parts = []

        if turn.text and turn.text.strip():
            text_parts.append(turn.text.strip())

        for elem in turn:
            if elem.tag not in SKIP_TAGS:
                if elem.text and elem.text.strip():
                    text_parts.append(elem.text.strip())
            if elem.tail and elem.tail.strip():
                text_parts.append(elem.tail.strip())

        text = " ".join(text_parts).strip()

        if not text or (end - start) < 0.1:
            continue

        segments.append({
            "speaker": speaker,
            "start": round(start, 3),
            "end": round(end, 3),
            "text": text,
        })

    segments.sort(key=lambda x: x["start"])
    return segments


def find_trs(file_id: str, trs_dir: str) -> str | None:
    """Find the TRS path for a given file_id. Returns None if not found."""
    import os
    path = os.path.join(trs_dir, f"{file_id}.trs")
    return path if os.path.exists(path) else None
```

---

## `utils/run_whisperx.py`

Run WhisperX on a single audio file. Models are loaded once and reused across calls.

```python
import os
import torch

# Lazy globals — loaded once on first call
_whisperx_model = None
_align_model = None
_align_meta = None
_diarize_model = None


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_models():
    global _whisperx_model, _align_model, _align_meta, _diarize_model
    if _whisperx_model is not None:
        return

    import whisperx

    device = _get_device()
    compute_type = "float16" if device == "cuda" else "int8"

    print(f"[WhisperX] Loading models on {device} ({compute_type})...")

    _whisperx_model = whisperx.load_model(
        "large-v3", device, compute_type=compute_type, language="sl"
    )

    # Slovenian alignment model may not exist — fall back to Croatian
    try:
        _align_model, _align_meta = whisperx.load_align_model(
            language_code="sl", device=device
        )
        print("[WhisperX] Using Slovenian alignment model")
    except Exception:
        print("[WhisperX] Slovenian alignment model not found, using Croatian (hr)")
        _align_model, _align_meta = whisperx.load_align_model(
            language_code="hr", device=device
        )

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("HF_TOKEN environment variable not set")

    _diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=hf_token, device=device
    )
    print("[WhisperX] Models ready.")


def run_whisperx(audio_path: str) -> list:
    """
    Run WhisperX on a single WAV file.
    Returns: [{"speaker": str, "start": float, "end": float, "text": str}, ...]
    """
    import whisperx

    _load_models()

    audio = whisperx.load_audio(audio_path)

    result = _whisperx_model.transcribe(audio, batch_size=16, language="sl")
    result = whisperx.align(
        result["segments"], _align_model, _align_meta, audio, _get_device(),
        return_char_alignments=False
    )
    diarize_segments = _diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    segments = []
    for seg in result["segments"]:
        segments.append({
            "speaker": seg.get("speaker", "SPEAKER_UNKNOWN"),
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "text": seg["text"].strip(),
        })

    return sorted(segments, key=lambda x: x["start"])
```

---

## `utils/run_our_pipeline.py`

**YOU MUST FILL IN `run_our_pipeline()` WITH YOUR ACTUAL PIPELINE CALL.**
Everything else (model caching, interface contract) stays the same.

```python
# Lazy global for any model/state your pipeline needs to keep loaded
_pipeline = None


def _load_pipeline():
    global _pipeline
    if _pipeline is not None:
        return
    # TODO: load your pipeline here (e.g. load model weights, initialize client)
    # Example:
    #   from my_pipeline import SpeakerDiarizationPipeline
    #   _pipeline = SpeakerDiarizationPipeline(device="cuda")
    print("[OurPipeline] TODO: replace _load_pipeline() with your actual loader")
    _pipeline = True  # placeholder


def run_our_pipeline(audio_path: str) -> list:
    """
    Run our custom pipeline on a single WAV file.
    Returns: [{"speaker": str, "start": float, "end": float, "text": str}, ...]

    FILL THIS IN with your actual pipeline call.
    """
    _load_pipeline()

    # TODO: replace with your real call, e.g.:
    #   result = _pipeline.process(audio_path)
    #   return [{"speaker": s.speaker, "start": s.start, "end": s.end, "text": s.text}
    #           for s in result.segments]

    raise NotImplementedError(
        "Fill in run_our_pipeline() in utils/run_our_pipeline.py"
    )
```

---

## `utils/evaluate_der.py`

Compute DER between a pipeline's output and the gold RTTM for one file.

```python
from pyannote.metrics.diarization import DiarizationErrorRate
from .rttm_io import load_rttm, segments_to_annotation


def evaluate_der(gold_rttm_path: str, hypothesis_segments: list, collar: float = 0.0) -> dict:
    """
    Compute DER and component metrics for one file.

    Args:
        gold_rttm_path: path to the reference .rttm file
        hypothesis_segments: pipeline output (list of dicts)
        collar: forgiveness collar in seconds (0.0 = strict, 0.25 = standard lenient)

    Returns dict with keys: der, miss, fa, confusion, total_duration
    """
    metric = DiarizationErrorRate(collar=collar, skip_overlap=False)

    ref = load_rttm(gold_rttm_path)
    hyp = segments_to_annotation(hypothesis_segments)

    detail = metric(ref, hyp, detailed=True)
    total = detail["total"]

    if total == 0:
        return {"der": 0.0, "miss": 0.0, "fa": 0.0, "confusion": 0.0, "total_duration": 0.0}

    return {
        "der":            detail["diarization error rate"],
        "miss":           detail["missed detection"] / total,
        "fa":             detail["false alarm"] / total,
        "confusion":      detail["confusion"] / total,
        "total_duration": total,
    }
```

---

## `utils/evaluate_wer.py`

Compute WER between a pipeline's text output and the reference TRS text.

```python
import re
from jiwer import wer as jiwer_wer


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def evaluate_wer(ref_segments: list, hyp_segments: list) -> dict:
    """
    Compute speaker-independent WER for one file.
    Both inputs are lists of {"start", "end", "text", ...} dicts.
    Segments are concatenated in time order before comparison.

    Returns dict with keys: wer, ref_words, hyp_words
    """
    ref_text = normalize(
        " ".join(s["text"] for s in sorted(ref_segments, key=lambda x: x["start"]))
    )
    hyp_text = normalize(
        " ".join(s["text"] for s in sorted(hyp_segments, key=lambda x: x["start"]))
    )

    if not ref_text:
        return {"wer": None, "ref_words": 0, "hyp_words": len(hyp_text.split())}

    return {
        "wer":       jiwer_wer(ref_text, hyp_text),
        "ref_words": len(ref_text.split()),
        "hyp_words": len(hyp_text.split()),
    }
```

---

## `benchmark_single.py`

Benchmark a single WAV file. Pass the WAV path as argument.

```python
#!/usr/bin/env python3
"""
Benchmark a single Artur-J file.

Usage:
    python benchmark_single.py <wav_path>

Example:
    python benchmark_single.py data/datasets/artur-j/wav/Artur-J-0001.wav

Looks for matching .rttm and .trs files automatically.
Results are printed to stdout and saved to results/<file_id>.json.
"""

import sys
import os
import json
import time

# Paths
DATA_ROOT = "data/datasets/artur-j"
WAV_DIR   = os.path.join(DATA_ROOT, "wav")
RTTM_DIR  = os.path.join(DATA_ROOT, "rttm")
TRS_DIR   = os.path.join(DATA_ROOT, "trs")
RESULTS_DIR = "benchmark/results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# Add benchmark/ to path so utils imports work
sys.path.insert(0, os.path.dirname(__file__))

from utils.rttm_io import find_rttm
from utils.trs_parser import parse_trs, find_trs
from utils.run_whisperx import run_whisperx
from utils.run_our_pipeline import run_our_pipeline
from utils.evaluate_der import evaluate_der
from utils.evaluate_wer import evaluate_wer


def benchmark_file(wav_path: str) -> dict:
    file_id = os.path.splitext(os.path.basename(wav_path))[0]
    print(f"\n{'='*60}")
    print(f"File: {file_id}")
    print(f"{'='*60}")

    # Find companion files
    rttm_path = find_rttm(file_id, RTTM_DIR)
    trs_path  = find_trs(file_id, TRS_DIR)

    if not rttm_path:
        print(f"  WARNING: no gold RTTM found for {file_id} — DER will be skipped")
    if not trs_path:
        print(f"  WARNING: no TRS found for {file_id} — WER will be skipped")

    # Parse reference text
    ref_segments = []
    if trs_path:
        try:
            ref_segments = parse_trs(trs_path)
        except Exception as e:
            print(f"  TRS parse error: {e}")

    result = {
        "file_id": file_id,
        "wav_path": wav_path,
        "whisperx": {},
        "our_pipeline": {},
    }

    # --- WhisperX ---
    print("\n[WhisperX] Running...")
    t0 = time.time()
    try:
        wx_segments = run_whisperx(wav_path)
        wx_time = time.time() - t0
        print(f"  Done in {wx_time:.1f}s ({len(wx_segments)} segments)")

        if rttm_path:
            for collar in [0.0, 0.25]:
                der = evaluate_der(rttm_path, wx_segments, collar=collar)
                result["whisperx"][f"der_collar{collar}"] = der
                print(f"  DER (collar={collar}): {der['der']*100:.2f}%  "
                      f"Miss={der['miss']*100:.2f}%  "
                      f"FA={der['fa']*100:.2f}%  "
                      f"Conf={der['confusion']*100:.2f}%")

        if ref_segments:
            wer = evaluate_wer(ref_segments, wx_segments)
            result["whisperx"]["wer"] = wer
            print(f"  WER: {wer['wer']*100:.2f}%  "
                  f"(ref={wer['ref_words']} words, hyp={wer['hyp_words']} words)")

        result["whisperx"]["segments"] = wx_segments
        result["whisperx"]["runtime_sec"] = wx_time

    except Exception as e:
        print(f"  ERROR: {e}")
        result["whisperx"]["error"] = str(e)

    # --- Our Pipeline ---
    print("\n[OurPipeline] Running...")
    t0 = time.time()
    try:
        op_segments = run_our_pipeline(wav_path)
        op_time = time.time() - t0
        print(f"  Done in {op_time:.1f}s ({len(op_segments)} segments)")

        if rttm_path:
            for collar in [0.0, 0.25]:
                der = evaluate_der(rttm_path, op_segments, collar=collar)
                result["our_pipeline"][f"der_collar{collar}"] = der
                print(f"  DER (collar={collar}): {der['der']*100:.2f}%  "
                      f"Miss={der['miss']*100:.2f}%  "
                      f"FA={der['fa']*100:.2f}%  "
                      f"Conf={der['confusion']*100:.2f}%")

        if ref_segments:
            wer = evaluate_wer(ref_segments, op_segments)
            result["our_pipeline"]["wer"] = wer
            print(f"  WER: {wer['wer']*100:.2f}%  "
                  f"(ref={wer['ref_words']} words, hyp={wer['hyp_words']} words)")

        result["our_pipeline"]["segments"] = op_segments
        result["our_pipeline"]["runtime_sec"] = op_time

    except Exception as e:
        print(f"  ERROR: {e}")
        result["our_pipeline"]["error"] = str(e)

    # --- Summary ---
    print(f"\n{'─'*40}")
    print("SUMMARY")

    wx_der  = result["whisperx"].get("der_collar0.0", {}).get("der")
    op_der  = result["our_pipeline"].get("der_collar0.0", {}).get("der")
    wx_wer  = result["whisperx"].get("wer", {}).get("wer")
    op_wer  = result["our_pipeline"].get("wer", {}).get("wer")

    if wx_der is not None and op_der is not None:
        delta = (op_der - wx_der) * 100
        winner = "OurPipeline" if op_der < wx_der else "WhisperX"
        print(f"  DER:  WhisperX={wx_der*100:.2f}%  OurPipeline={op_der*100:.2f}%  "
              f"Δ={delta:+.2f}%  → {winner} wins")

    if wx_wer is not None and op_wer is not None:
        delta = (op_wer - wx_wer) * 100
        winner = "OurPipeline" if op_wer < wx_wer else "WhisperX"
        print(f"  WER:  WhisperX={wx_wer*100:.2f}%  OurPipeline={op_wer*100:.2f}%  "
              f"Δ={delta:+.2f}%  → {winner} wins")

    # Save result
    out_path = os.path.join(RESULTS_DIR, f"{file_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {out_path}")

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmark_single.py <wav_path>")
        sys.exit(1)

    wav_path = sys.argv[1]
    if not os.path.exists(wav_path):
        print(f"ERROR: file not found: {wav_path}")
        sys.exit(1)

    benchmark_file(wav_path)
```

---

## `benchmark_loop.py`

Runs continuously through the dataset, one file at a time, until Ctrl+C.
Skips files that already have a result saved. Prints a running aggregate after each file.

```python
#!/usr/bin/env python3
"""
Continuously benchmark Artur-J files until Ctrl+C.

Usage:
    python benchmark_loop.py [--shuffle]

Options:
    --shuffle    Process files in random order instead of alphabetical

Files that already have a saved result in benchmark/results/ are skipped.
Press Ctrl+C at any time to stop cleanly — partial results are always saved.
A running aggregate is printed after each file.
"""

import sys
import os
import json
import random
import signal

DATA_ROOT   = "data/datasets/artur-j"
WAV_DIR     = os.path.join(DATA_ROOT, "wav")
RESULTS_DIR = "benchmark/results"

os.makedirs(RESULTS_DIR, exist_ok=True)
sys.path.insert(0, os.path.dirname(__file__))

from benchmark_single import benchmark_file


def get_wav_files(shuffle: bool) -> list:
    files = sorted([
        os.path.join(WAV_DIR, f)
        for f in os.listdir(WAV_DIR)
        if f.endswith(".wav")
    ])
    if shuffle:
        random.shuffle(files)
    return files


def already_done(wav_path: str) -> bool:
    file_id = os.path.splitext(os.path.basename(wav_path))[0]
    return os.path.exists(os.path.join(RESULTS_DIR, f"{file_id}.json"))


def print_aggregate():
    """Read all saved results and print running aggregate metrics."""
    results = []
    for fname in os.listdir(RESULTS_DIR):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(RESULTS_DIR, fname)) as f:
            try:
                results.append(json.load(f))
            except Exception:
                pass

    if not results:
        return

    def avg(key_path):
        vals = []
        for r in results:
            v = r
            for k in key_path:
                v = v.get(k, {}) if isinstance(v, dict) else None
                if v is None:
                    break
            if isinstance(v, (int, float)):
                vals.append(v)
        return sum(vals) / len(vals) if vals else None

    print(f"\n{'━'*60}")
    print(f"RUNNING AGGREGATE  ({len(results)} files completed)")
    print(f"{'━'*60}")

    for system in ["whisperx", "our_pipeline"]:
        label = "WhisperX" if system == "whisperx" else "OurPipeline"
        der   = avg([system, "der_collar0.0", "der"])
        miss  = avg([system, "der_collar0.0", "miss"])
        fa    = avg([system, "der_collar0.0", "fa"])
        conf  = avg([system, "der_collar0.0", "confusion"])
        wer   = avg([system, "wer", "wer"])
        rt    = avg([system, "runtime_sec"])

        parts = [f"  {label}:"]
        if der  is not None: parts.append(f"DER={der*100:.2f}%")
        if miss is not None: parts.append(f"Miss={miss*100:.2f}%")
        if fa   is not None: parts.append(f"FA={fa*100:.2f}%")
        if conf is not None: parts.append(f"Conf={conf*100:.2f}%")
        if wer  is not None: parts.append(f"WER={wer*100:.2f}%")
        if rt   is not None: parts.append(f"avg_runtime={rt:.0f}s")
        print("  ".join(parts))

    print(f"{'━'*60}\n")


# Clean Ctrl+C handler
_stop = False
def _handle_sigint(sig, frame):
    global _stop
    print("\n\nCtrl+C received — finishing current file then stopping...")
    _stop = True

signal.signal(signal.SIGINT, _handle_sigint)


if __name__ == "__main__":
    shuffle = "--shuffle" in sys.argv
    wav_files = get_wav_files(shuffle)

    pending = [f for f in wav_files if not already_done(f)]
    skipped = len(wav_files) - len(pending)

    print(f"Found {len(wav_files)} WAV files.")
    print(f"  Already done: {skipped}")
    print(f"  To process:   {len(pending)}")
    if shuffle:
        print("  Order: random")
    print("\nPress Ctrl+C to stop cleanly at any time.\n")

    for i, wav_path in enumerate(pending):
        if _stop:
            break

        file_id = os.path.splitext(os.path.basename(wav_path))[0]
        print(f"\nFile {i+1}/{len(pending)}: {file_id}")

        try:
            benchmark_file(wav_path)
        except Exception as e:
            print(f"  UNHANDLED ERROR on {file_id}: {e}")
            # Continue to next file regardless

        print_aggregate()

    print("Loop finished." if not _stop else "Stopped by user.")
```

---

## Pre-flight Checks

Before running any benchmark, verify these things:

**1. Filename consistency between wav, rttm, trs:**
```bash
ls data/datasets/artur-j/wav/  | sed 's/.wav//'  | sort > /tmp/wav_ids.txt
ls data/datasets/artur-j/rttm/ | sed 's/.rttm//' | sort > /tmp/rttm_ids.txt
ls data/datasets/artur-j/trs/  | sed 's/.trs//'  | sort > /tmp/trs_ids.txt
diff /tmp/wav_ids.txt /tmp/rttm_ids.txt
diff /tmp/wav_ids.txt /tmp/trs_ids.txt
```
Any diff = mismatch that needs fixing before evaluation will work.

**2. Check a gold RTTM looks correct:**
```bash
head -10 data/datasets/artur-j/rttm/SOME_FILE.rttm
```
Expected format: `SPEAKER <file_id> 1 <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>`

**3. Spot-check TRS parsing on one file:**
After building the scripts, run this quick check (does not run any pipeline):
```python
from benchmark.utils.trs_parser import parse_trs
segs = parse_trs("data/datasets/artur-j/trs/SOME_FILE.trs")
for s in segs[:5]:
    print(s)
```
Confirm text and timestamps look sensible before running the full benchmark.

**4. Fill in your pipeline interface:**
Edit `benchmark/utils/run_our_pipeline.py` — replace the `NotImplementedError`
in `run_our_pipeline()` with your actual call.

**5. Set HF_TOKEN:**
```bash
export HF_TOKEN=your_huggingface_token
```

---

## Usage

```bash
# Benchmark one specific file
python benchmark/benchmark_single.py data/datasets/artur-j/wav/Artur-J-0001.wav

# Run continuously in alphabetical order until Ctrl+C
python benchmark/benchmark_loop.py

# Run continuously in random order
python benchmark/benchmark_loop.py --shuffle
```

Results are saved to `benchmark/results/<file_id>.json` after each file.
The loop script skips already-completed files, so you can stop and resume freely.