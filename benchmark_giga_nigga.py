import os
import time
import torch
import soundfile as sf
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
from pyannote.metrics.detection import DetectionErrorRate
from pyannote.database.util import load_rttm
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HF_TOKEN")

DATASET_TIME_LIMIT = 2700 # In seconds

# ── M O D E L S ───────────────────────────────────────────────────────────────
MODELS = [
    "pyannote/speaker-diarization-community-1",
    "pyannote/speaker-diarization-3.1",
]

# ── D A T A S E T S ───────────────────────────────────────────────────────────
DATASETS_ROOT = "datasets"

datasets = ["VoxConverse"]

# ── B U C K E T S   B Y   A U D I O   L E N G T H ────────────────────────────
def get_duration_bucket(duration_seconds):
    if duration_seconds < 60:
        return "short (<1min)"
    elif duration_seconds < 300:
        return "medium (1-5min)"
    elif duration_seconds < 900:
        return "long (5-15min)"
    else:
        return "very long (>15min)"

# ── B E N C H M A R K ─────────────────────────────────────────────────────────
for model_name in MODELS:
    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    print(f"{'='*60}")

    pipeline = Pipeline.from_pretrained(model_name, token=token)
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    model_short = model_name.split("/")[-1]

    for dataset in datasets:
        audio_dir   = os.path.join(DATASETS_ROOT, dataset, "wav")
        rttm_dir    = os.path.join(DATASETS_ROOT, dataset, "rttm")
        result_file = os.path.join(DATASETS_ROOT, dataset, f"result_{model_short}.txt")

        if not os.path.isdir(audio_dir) or not os.path.isdir(rttm_dir):
            print(f"Skipping {dataset} — missing wav/ or rttm/ folder")
            continue

        print(f"\nDataset: {dataset}")

        # Metrics accumulators
        der_metric = DiarizationErrorRate()
        jer_metric = JaccardErrorRate()
        det_metric = DetectionErrorRate()  # gives miss + false alarm

        # Per-file records for length analysis
        records = []

        dataset_start = time.time()
        timed_out = False

        for audio_file in os.listdir(audio_dir):
            if not audio_file.endswith(".wav"):
                continue

            #   T I M E O U T   C H E C K
            elapsed = time.time() - dataset_start
            if elapsed >= DATASET_TIME_LIMIT:
                print(f"  ⏱  Time limit reached ({DATASET_TIME_LIMIT/3600:.1f}h) — stopping early.")
                timed_out = True
                break

            file_name  = os.path.splitext(audio_file)[0]
            audio_path = os.path.join(audio_dir, audio_file)
            rttm_path  = os.path.join(rttm_dir, file_name + ".rttm")

            if not os.path.exists(rttm_path):
                print(f"  No RTTM for {file_name}, skipping...")
                continue

            # Get audio duration
            info = sf.info(audio_path)
            duration = info.duration

            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - dataset_start))
            print(f"  Processing {file_name} ({duration:.1f}s) [elapsed {elapsed_str}]...")

            with ProgressHook() as hook:
                output = pipeline(audio_path, hook=hook)

            hypothesis = output.speaker_diarization
            reference  = load_rttm(rttm_path)[file_name]

            # Compute metrics
            der      = der_metric(reference, hypothesis)
            jer      = jer_metric(reference, hypothesis)
            det      = det_metric(reference, hypothesis)

            # Extract miss/false alarm/confusion from DER components
            detail       = der_metric(reference, hypothesis, detailed=True)
            miss         = detail["missed detection"] / detail["total"] if detail["total"] > 0 else 0
            false_alarm  = detail["false alarm"] / detail["total"] if detail["total"] > 0 else 0
            confusion    = detail["confusion"] / detail["total"] if detail["total"] > 0 else 0

            records.append({
                "file":        file_name,
                "duration":    duration,
                "bucket":      get_duration_bucket(duration),
                "der":         der,
                "jer":         jer,
                "miss":        miss,
                "false_alarm": false_alarm,
                "confusion":   confusion,
            })

            print(f"    DER={der:.3f}  JER={jer:.3f}  Miss={miss:.3f}  FA={false_alarm:.3f}  Conf={confusion:.3f}")

        total_elapsed = time.time() - dataset_start

         #   W R I T E   R E S U L T S
        with open(result_file, "w", buffering=1) as result:
            result.write(f"Model: {model_name}\n")
            result.write(f"Dataset: {dataset}\n")
            if timed_out:
                result.write(f"NOTE: Run stopped early — time limit of {DATASET_TIME_LIMIT/3600:.1f}h reached.\n")
                result.write(f"      Only {len(records)} file(s) were processed.\n")
            result.write(f"Total elapsed: {time.strftime('%H:%M:%S', time.gmtime(total_elapsed))}\n")
            result.write("=" * 60 + "\n\n")
 
            # Per-file results
            result.write("PER FILE:\n")
            result.write(f"{'File':<30} {'Dur(s)':>7} {'DER':>7} {'JER':>7} {'Miss':>7} {'FA':>7} {'Conf':>7}\n")
            result.write("-" * 75 + "\n")
            for r in records:
                result.write(
                    f"{r['file']:<30} {r['duration']:>7.1f} {r['der']:>7.3f} "
                    f"{r['jer']:>7.3f} {r['miss']:>7.3f} {r['false_alarm']:>7.3f} {r['confusion']:>7.3f}\n"
                )
 
            # Overall averages
            if records:
                result.write("\n" + "=" * 60 + "\n")
                result.write("OVERALL AVERAGES:\n")
                for metric in ["der", "jer", "miss", "false_alarm", "confusion"]:
                    avg = sum(r[metric] for r in records) / len(records)
                    result.write(f"  Avg {metric.upper():<12} = {avg:.3f}\n")
 
            # Per bucket analysis
            result.write("\n" + "=" * 60 + "\n")
            result.write("BY AUDIO LENGTH:\n")
            buckets = {}
            for r in records:
                buckets.setdefault(r["bucket"], []).append(r)
 
            for bucket, bucket_records in sorted(buckets.items()):
                result.write(f"\n  {bucket} — {len(bucket_records)} file(s)\n")
                for metric in ["der", "jer", "miss", "false_alarm", "confusion"]:
                    avg = sum(r[metric] for r in bucket_records) / len(bucket_records)
                    result.write(f"    Avg {metric.upper():<12} = {avg:.3f}\n")
 
        status = "⏱  partial (timed out)" if timed_out else "✓ complete"
        print(f"  Results saved to {result_file} [{status}]")

print("\nDone!")