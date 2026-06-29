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

audio_files = sorted(f for f in AUDIO_DIR.glob("*.wav") if not f.name.startswith("."))[:MAX_FILES]


def run_benchmark(model: DiarizationModel) -> list[dict]:
    der_metric = DiarizationErrorRate()
    jer_metric = JaccardErrorRate()
    results = []

    for i, audio_path in enumerate(audio_files):
        uri = audio_path.stem
        rttm_path = RTTM_DIR / f"{uri}.rttm"
        if not rttm_path.exists():
            print(f"    [{i+1}] {uri}: no RTTM, skipping")
            continue

        diarization = model.diarize(str(audio_path), preprocess=False)
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


def format_results(results: list[dict]) -> list[str]:
    lines = [f"{'File':<20} {'DER':>10} {'JER':>10} {'Miss':>10} {'FA':>10} {'Conf':>10}", "-" * 72]
    for r in results:
        lines.append(f"{r['uri']:<20} {r['der']:>10.4f} {r['jer']:>10.4f} {r['miss']:>10.4f} {r['false_alarm']:>10.4f} {r['confusion']:>10.4f}")

    if not results:
        return lines

    lines += ["", "AVERAGES:"]
    for key, label in [("der", "DER"), ("jer", "JER"), ("miss", "Miss"), ("false_alarm", "False Alarm"), ("confusion", "Confusion")]:
        lines.append(f"  {label:<15} {avg(results, key):.4f}")

    return lines


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = OUTPUT_DIR / f"benchmark_{timestamp}.txt"

    print(f"Files:  {len(audio_files)}  |  Output: {result_file}\n")

    with open(result_file, "w") as out:
        for model_name in MODELS:
            print(f"\n{'='*60}")
            print(f"Model: {model_name}")
            print(f"{'='*60}")

            model = DiarizationModel(asr=None, pyannote_model=model_name)
            results = run_benchmark(model)

            block = [f"Model: {model_name}", "=" * 72] + format_results(results)

            print()
            print("\n".join(block))

            out.write("\n".join(block) + "\n\n")
            out.flush()

    print(f"\nSaved to {result_file}")
