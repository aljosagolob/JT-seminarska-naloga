import warnings
warnings.filterwarnings("ignore")

import os
import torch
from pathlib import Path
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
from preprocess import load_audio, preprocess

load_dotenv()

AUDIO_DIR   = Path("datasets/VoxConverse/wav")
RTTM_DIR    = Path("datasets/VoxConverse/rttm")
RESULT_FILE = Path("datasets/VoxConverse/comparison_result.txt")

MAX_FILES = None  # set to an int to limit, None = all files

MODELS = [
    "pyannote/speaker-diarization-3.1",
    "pyannote/speaker-diarization-community-1",
]

# Best params from trial #243 (optimized on community-1)
BEST_PARAMS = {
    "speed":              1.0000,
    "highpass_cutoff":  289.9122,
    "lowpass_cutoff":  6070.9890,
    "gain_db":           -5.1013,
    "target_lufs":      -25.6178,
    "eq_center_freq":  3065.4163,
    "eq_gain_db":        10.5143,
    "eq_q":               2.9523,
    "comp_threshold_db": -17.6729,
    "comp_ratio":         2.3863,
    "noise_reduce":       0.0559,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
token  = os.getenv("HF_TOKEN")

audio_files = sorted(f for f in AUDIO_DIR.glob("*.wav") if not f.name.startswith("."))[:MAX_FILES]
print(f"Found {len(audio_files)} files, device={device}\n")


def run_pass(pipeline, label: str, params: dict | None) -> list[dict]:
    print(f"  --- {label} ---")
    der_metric = DiarizationErrorRate()
    jer_metric = JaccardErrorRate()
    results = []
    for i, audio_path in enumerate(audio_files):
        uri = audio_path.stem
        rttm_path = RTTM_DIR / f"{uri}.rttm"
        if not rttm_path.exists():
            print(f"    [{i+1}] {uri}: no RTTM, skipping")
            continue

        if params is None:
            # Identical to benchmark_giga_nigga: pass file path directly
            output = pipeline(str(audio_path))
        else:
            waveform, sample_rate = load_audio(str(audio_path))
            waveform, sample_rate = preprocess(waveform, sample_rate, params)
            output = pipeline({"waveform": waveform, "sample_rate": sample_rate})

        reference  = load_rttm(str(rttm_path))[uri]
        hypothesis = output.speaker_diarization

        detail      = der_metric(reference, hypothesis, detailed=True)  # type: ignore[assignment]
        der         = detail["diarization error rate"]
        total       = detail["total"]
        miss        = detail["missed detection"] / total if total > 0 else 0.0
        false_alarm = detail["false alarm"]      / total if total > 0 else 0.0
        confusion   = detail["confusion"]        / total if total > 0 else 0.0
        jer         = jer_metric(reference, hypothesis)

        print(f"    [{i+1}/{len(audio_files)}] {uri}: DER={der:.4f}  JER={jer:.4f}  Miss={miss:.4f}  FA={false_alarm:.4f}  Conf={confusion:.4f}")
        results.append({
            "uri":         uri,
            "der":         der,
            "jer":         jer,
            "miss":        miss,
            "false_alarm": false_alarm,
            "confusion":   confusion,
        })
    return results


def avg(records, key):
    return sum(r[key] for r in records) / len(records) if records else 0.0


def format_table(raw_results, prep_results) -> list[str]:
    raw_map  = {r["uri"]: r for r in raw_results}
    prep_map = {r["uri"]: r for r in prep_results}
    common   = [uri for uri in raw_map if uri in prep_map]

    lines = []

    # Per-file DER comparison
    lines.append(f"{'File':<20} {'Raw DER':>10} {'Prep DER':>10} {'Delta':>10} {'Better?':>10}")
    lines.append("-" * 64)
    for uri in common:
        r     = raw_map[uri]["der"]
        p     = prep_map[uri]["der"]
        delta = p - r
        better = "prep" if delta < 0 else ("raw" if delta > 0 else "tie")
        lines.append(f"{uri:<20} {r:>10.4f} {p:>10.4f} {delta:>+10.4f} {better:>10}")

    if not common:
        return lines

    lines.append("")
    lines.append("AVERAGES:")
    lines.append(f"{'Metric':<15} {'Raw':>10} {'Preprocessed':>14} {'Delta':>10}")
    lines.append("-" * 52)
    for key, label in [("der", "DER"), ("jer", "JER"), ("miss", "Miss"), ("false_alarm", "False Alarm"), ("confusion", "Confusion")]:
        r_avg = avg([raw_map[u]  for u in common], key)
        p_avg = avg([prep_map[u] for u in common], key)
        delta = p_avg - r_avg
        lines.append(f"{label:<15} {r_avg:>10.4f} {p_avg:>14.4f} {delta:>+10.4f}")

    lines.append("")
    avg_der_raw  = avg([raw_map[u]  for u in common], "der")
    avg_der_prep = avg([prep_map[u] for u in common], "der")
    lines.append("Preprocessing HELPS" if avg_der_prep < avg_der_raw else "Preprocessing HURTS (or ties)")

    return lines


RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)

for model_name in MODELS:
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    pipeline = Pipeline.from_pretrained(model_name, token=token)
    assert pipeline is not None
    pipeline.to(device)

    raw_results  = run_pass(pipeline, "Raw (no preprocessing)", None)
    print()
    prep_results = run_pass(pipeline, "Preprocessed (trial #243)", BEST_PARAMS)

    table = format_table(raw_results, prep_results)
    block = [f"Model: {model_name}", "=" * 64] + table

    print()
    print("\n".join(block))

    with open(RESULT_FILE, "a") as f:
        f.write("\n".join(block) + "\n\n")
        f.flush()
    print(f"Saved to {RESULT_FILE}")
