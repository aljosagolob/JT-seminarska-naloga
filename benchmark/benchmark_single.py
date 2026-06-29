#!/usr/bin/env python3
"""
Benchmark a single Artur-J file.

Usage:
    python benchmark/benchmark_single.py <wav_path>

Example:
    python benchmark/benchmark_single.py datasets/artur-j/wav/Artur-J-Gvecg-P500030-avd.wav

Looks for matching .rttm and .trs files automatically.
Results are printed to stdout and saved to benchmark/results/<file_id>.json.
"""

import sys
import os
import json
import time
from dotenv import load_dotenv
load_dotenv()

# SpeechBrain has optional integrations (k2, nlp, etc.) that crash on lazy import
# when the underlying package isn't installed. Pre-stub every known one.
import sys, types

def _stub_speechbrain_integrations():
    import importlib, pkgutil
    try:
        import speechbrain.integrations as _sb_int
        for _finder, _name, _ispkg in pkgutil.iter_modules(_sb_int.__path__):
            full = f"speechbrain.integrations.{_name}"
            if full not in sys.modules:
                try:
                    importlib.import_module(full)
                except Exception:
                    sys.modules[full] = types.ModuleType(full)
    except Exception:
        pass

_stub_speechbrain_integrations()

# PyTorch 2.6+ defaults weights_only=True which breaks pyannote checkpoints.
# Override torch.load globally to restore the old behaviour for all model loading.
import torch
_torch_load_orig = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs["weights_only"] = False
    return _torch_load_orig(*args, **kwargs)
torch.load = _torch_load_compat

# pyannote was designed for older torchaudio — patch missing API before any import
import importlib
import torchaudio

# 1. AudioMetaData
if not hasattr(torchaudio, "AudioMetaData"):
    _found = False
    for _path in (
        "torchaudio.backend.common",
        "torchaudio._backend.utils",
        "torchaudio._backend.common",
        "torchaudio.io",
        "torchaudio.io._compat",
    ):
        try:
            _m = importlib.import_module(_path)
            if hasattr(_m, "AudioMetaData"):
                torchaudio.AudioMetaData = _m.AudioMetaData
                _found = True
                break
        except (ImportError, ModuleNotFoundError):
            continue
    if not _found:
        from dataclasses import dataclass
        @dataclass
        class _AudioMetaData:
            sample_rate: int = 0
            num_frames: int = 0
            num_channels: int = 0
            bits_per_sample: int = 0
            encoding: str = ""
        torchaudio.AudioMetaData = _AudioMetaData

# 2. list_audio_backends
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

# 3. get_audio_backend / set_audio_backend
if not hasattr(torchaudio, "get_audio_backend"):
    torchaudio.get_audio_backend = lambda: "soundfile"
if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda backend: None

# 4. info (used by some pyannote versions to probe files)
if not hasattr(torchaudio, "info"):
    def _torchaudio_info(path, format=None):
        import soundfile as sf
        info = sf.info(path)
        return torchaudio.AudioMetaData(
            sample_rate=info.samplerate,
            num_frames=info.frames,
            num_channels=info.channels,
            bits_per_sample=16,
            encoding="PCM_S",
        )
    torchaudio.info = _torchaudio_info

DATA_ROOT   = "datasets/artur-j"
RTTM_DIR    = os.path.join(DATA_ROOT, "rttm")
TRS_DIR     = os.path.join(DATA_ROOT, "trs")
RESULTS_DIR = "benchmark/results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# Add benchmark/ to path so utils.* imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

    rttm_path = find_rttm(file_id, RTTM_DIR)
    trs_path  = find_trs(file_id, TRS_DIR)

    if not rttm_path:
        print(f"  WARNING: no gold RTTM found for {file_id} — DER will be skipped")
    if not trs_path:
        print(f"  WARNING: no TRS found for {file_id} — WER will be skipped")

    ref_segments = []
    if trs_path:
        try:
            ref_segments = parse_trs(trs_path)
        except Exception as e:
            print(f"  TRS parse error: {e}")

    result = {
        "file_id":      file_id,
        "wav_path":     wav_path,
        "whisperx":     {},
        "our_pipeline": {},
    }

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
            if wer["wer"] is not None:
                print(f"  WER: {wer['wer']*100:.2f}%  "
                      f"(ref={wer['ref_words']} words, hyp={wer['hyp_words']} words)")

        result["our_pipeline"]["segments"]    = op_segments
        result["our_pipeline"]["runtime_sec"] = op_time

    except Exception as e:
        print(f"  ERROR: {e}")
        result["our_pipeline"]["error"] = str(e)

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
            if wer["wer"] is not None:
                print(f"  WER: {wer['wer']*100:.2f}%  "
                      f"(ref={wer['ref_words']} words, hyp={wer['hyp_words']} words)")

        result["whisperx"]["segments"]    = wx_segments
        result["whisperx"]["runtime_sec"] = wx_time

    except Exception as e:
        print(f"  ERROR: {e}")
        result["whisperx"]["error"] = str(e)

    # --- Summary ---
    print(f"\n{'─'*40}")
    print("SUMMARY")

    wx_der = result["whisperx"].get("der_collar0.0", {}).get("der")
    op_der = result["our_pipeline"].get("der_collar0.0", {}).get("der")
    wx_wer = result["whisperx"].get("wer", {}).get("wer")
    op_wer = result["our_pipeline"].get("wer", {}).get("wer")

    if wx_der is not None and op_der is not None:
        delta  = (op_der - wx_der) * 100
        winner = "OurPipeline" if op_der < wx_der else "WhisperX"
        print(f"  DER:  WhisperX={wx_der*100:.2f}%  OurPipeline={op_der*100:.2f}%  "
              f"Δ={delta:+.2f}%  → {winner} wins")

    if wx_wer is not None and op_wer is not None:
        delta  = (op_wer - wx_wer) * 100
        winner = "OurPipeline" if op_wer < wx_wer else "WhisperX"
        print(f"  WER:  WhisperX={wx_wer*100:.2f}%  OurPipeline={op_wer*100:.2f}%  "
              f"Δ={delta:+.2f}%  → {winner} wins")

    out_path = os.path.join(RESULTS_DIR, f"{file_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {out_path}")

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmark/benchmark_single.py <wav_path>")
        sys.exit(1)

    wav_path = sys.argv[1]
    if not os.path.exists(wav_path):
        print(f"ERROR: file not found: {wav_path}")
        sys.exit(1)

    benchmark_file(wav_path)
