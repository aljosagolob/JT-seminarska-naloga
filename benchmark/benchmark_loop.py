#!/usr/bin/env python3
"""
Continuously benchmark Artur-J files until Ctrl+C.

Usage:
    python benchmark/benchmark_loop.py [--shuffle]

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

DATA_ROOT   = "datasets/artur-j"
WAV_DIR     = os.path.join(DATA_ROOT, "wav")
RESULTS_DIR = "benchmark/results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# Add benchmark/ to path so benchmark_single and utils.* imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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


def _handle_sigint(sig, frame):
    print("\n\nCtrl+C — exiting now.")
    os._exit(0)

signal.signal(signal.SIGINT, _handle_sigint)


if __name__ == "__main__":
    shuffle   = "--shuffle" in sys.argv
    wav_files = get_wav_files(shuffle)
    pending   = [f for f in wav_files if not already_done(f)]
    skipped   = len(wav_files) - len(pending)

    print(f"Found {len(wav_files)} WAV files.")
    print(f"  Already done: {skipped}")
    print(f"  To process:   {len(pending)}")
    if shuffle:
        print("  Order: random")
    print("\nPress Ctrl+C to stop cleanly at any time.\n")

    for i, wav_path in enumerate(pending):
        file_id = os.path.splitext(os.path.basename(wav_path))[0]
        print(f"\nFile {i+1}/{len(pending)}: {file_id}")

        try:
            benchmark_file(wav_path)
        except Exception as e:
            print(f"  UNHANDLED ERROR on {file_id}: {e}")

        print_aggregate()

    print("Loop finished.")
