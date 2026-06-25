import warnings
warnings.filterwarnings("ignore")

import sys
from datetime import datetime
from pathlib import Path

from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import DiarizationModel

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
AUDIO_DIR = Path("datasets/VoxConverse/wav")
RTTM_DIR = Path("datasets/VoxConverse/rttm")
OUTPUT_DIR = Path("datasets/VoxConverse")

# ── Config ────────────────────────────────────────────────────────────────────
MAX_FILES = 10   # None = all files

MODELS = [
    "pyannote/speaker-diarization-3.1",
    "pyannote/speaker-diarization-community-1",
]

BEST_PARAMS = {
    # Acoustic preprocessing
    "speed": 1.0,
    "highpass_cutoff": 289.9122,
    "lowpass_cutoff": 6070.9890,
    "target_lufs": -25.6178,
    "eq_gain_db": 10.5143,
    "comp_threshold_db": -17.6729,
    "comp_ratio": 2.3863,

    # Denoising
    "denoise": True,
    "noise_reduce_strength": 0.75,
    "gate_threshold_db": -30.0,
    "denoise_comp_threshold_db": -16.0,
    "denoise_comp_ratio": 4.0,
    "low_shelf_gain_db": 10.0,
    "denoise_gain_db": 2.0,
}

audio_files = sorted(f for f in AUDIO_DIR.glob("*.wav") if not f.name.startswith("."))[:MAX_FILES]


def run_pass(model: DiarizationModel, label: str, preprocess: bool = True) -> list[dict]:
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

        diarization = model.diarize(str(audio_path), preprocess=preprocess)
        reference = load_rttm(str(rttm_path))[uri]
        hypothesis = diarization.speaker_diarization

        detail = der_metric(reference, hypothesis, detailed=True)  # type: ignore
        der = detail["diarization error rate"]
        total = detail["total"]
        miss = detail["missed detection"] / total if total > 0 else 0.0
        false_alarm = detail["false alarm"] / total if total > 0 else 0.0
        confusion = detail["confusion"] / total if total > 0 else 0.0
        jer = jer_metric(reference, hypothesis)

        print(f"    [{i+1}/{len(audio_files)}] {uri}: DER={der:.4f}  JER={jer:.4f}  Miss={miss:.4f}  FA={false_alarm:.4f}  Conf={confusion:.4f}")
        results.append({"uri": uri, "der": der, "jer": jer, "miss": miss, "false_alarm": false_alarm, "confusion": confusion})

    return results


def avg(records: list[dict], key: str) -> float:
    return sum(r[key] for r in records) / len(records) if records else 0.0


def format_table(raw_results: list[dict], prep_results: list[dict]) -> list[str]:
    raw_map = {r["uri"]: r for r in raw_results}
    prep_map = {r["uri"]: r for r in prep_results}
    common = [uri for uri in raw_map if uri in prep_map]

    lines = [f"{'File':<20} {'Raw DER':>10} {'Prep DER':>10} {'Delta':>10} {'Better?':>10}", "-" * 64]
    for uri in common:
        r, p = raw_map[uri]["der"], prep_map[uri]["der"]
        delta = p - r
        lines.append(f"{uri:<20} {r:>10.4f} {p:>10.4f} {delta:>+10.4f} {'prep' if delta < 0 else ('raw' if delta > 0 else 'tie'):>10}")

    if not common:
        return lines

    lines += ["", "AVERAGES:", f"{'Metric':<15} {'Raw':>10} {'Preprocessed':>14} {'Delta':>10}", "-" * 52]
    for key, label in [("der", "DER"), ("jer", "JER"), ("miss", "Miss"), ("false_alarm", "False Alarm"), ("confusion", "Confusion")]:
        r_avg = avg([raw_map[u] for u in common], key)
        p_avg = avg([prep_map[u] for u in common], key)
        lines.append(f"{label:<15} {r_avg:>10.4f} {p_avg:>14.4f} {p_avg - r_avg:>+10.4f}")

    lines.append("")
    avg_raw = avg([raw_map[u] for u in common], "der")
    avg_prep = avg([prep_map[u] for u in common], "der")
    lines.append("Preprocessing HELPS" if avg_prep < avg_raw else "Preprocessing HURTS (or ties)")
    return lines


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = OUTPUT_DIR / f"comparison_{timestamp}.txt"

    print(f"Files:  {len(audio_files)}  |  Output: {result_file}\n")

    with open(result_file, "w") as out:
        for model_name in MODELS:
            print(f"\n{'='*60}")
            print(f"Model: {model_name}")
            print(f"{'='*60}")

            model = DiarizationModel(asr=None, pyannote_model=model_name)

            raw_results = run_pass(model, "Raw (no preprocessing)", preprocess=False)

            print()
            model.set_params(BEST_PARAMS)
            prep_results = run_pass(model, "Preprocessed (best params)")

            table = format_table(raw_results, prep_results)
            block = [f"Model: {model_name}", "=" * 64] + table

            print()
            print("\n".join(block))

            out.write("\n".join(block) + "\n\n")
            out.flush()

    print(f"\nSaved to {result_file}")
