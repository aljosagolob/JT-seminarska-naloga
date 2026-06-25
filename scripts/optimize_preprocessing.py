import warnings
warnings.filterwarnings("ignore")

import json
import sys
from datetime import datetime
from pathlib import Path

import optuna
from dotenv import load_dotenv
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import DiarizationModel

load_dotenv()

# Paths — pick the most recent chunk run automatically
_CHUNKS_BASE = Path("datasets/artur-j/chunks")
_latest = max(_CHUNKS_BASE.iterdir(), key=lambda p: p.name) if _CHUNKS_BASE.exists() else None
AUDIO_DIR = _latest / "audio" if _latest else Path("datasets/artur-j/chunks/latest/audio")
RTTM_DIR  = _latest / "rttm"  if _latest else Path("datasets/artur-j/chunks/latest/rttm")
OUTPUT_DIR = Path("results")

# Run configuration
N_EVAL_FILES = 10      # audio files used per trial (None = all)
N_TRIALS = 50
STUDY_NAME = "diarization_preprocessing"
STORAGE = f"sqlite:///{OUTPUT_DIR / 'optuna.db'}"
OPTIMIZE_DENOISE = True  # include denoise params in search space

# Search space (min, max) 
PARAM_RANGES = {
    # Acoustic preprocessing
    "speed": (0.33, 3),
    "highpass_cutoff": (0.0, 500.0),
    "lowpass_cutoff": (2000.0, 12000.0),
    "target_lufs": (-32.0, -16.0),
    "eq_gain_db": (-6.0, 12.0),
    "comp_threshold_db": (-40.0, -10.0),
    "comp_ratio": (1.0, 8.0),

    # Denoising
    "noise_reduce_strength": (0.0, 1.0),
    "gate_threshold_db": (-50.0, -10.0),
    "denoise_comp_threshold_db": (-40.0, -10.0),
    "denoise_comp_ratio": (1.0, 8.0),
    "low_shelf_gain_db": (0.0, 15.0),
    "denoise_gain_db": (-3.0, 6.0),
}

model = DiarizationModel(asr=None)


def evaluate(params: dict) -> float:
    model.set_params(params)
    der_metric = DiarizationErrorRate()
    audio_files = sorted(f for f in AUDIO_DIR.glob("*.wav") if not f.name.startswith("."))[:N_EVAL_FILES]

    for i, audio_path in enumerate(audio_files):
        uri = audio_path.stem
        rttm_path = RTTM_DIR / f"{uri}.rttm"

        print(f"  [{i+1}/{len(audio_files)}] {uri}", end="", flush=True)

        diarization = model.diarize(str(audio_path))
        reference = load_rttm(str(rttm_path))[uri]
        der_metric(reference, diarization.speaker_diarization, uem=None)

        print(f" DER={float(abs(der_metric)):.4f}")

    return float(abs(der_metric))


# Optimization objective function
def objective(trial: optuna.Trial) -> float:
    def suggest(name: str) -> float:
        lo, hi = PARAM_RANGES[name]
        return trial.suggest_float(name, lo, hi)

    params = {
        # Acoustic preprocessing
        "speed":             suggest("speed"),
        "highpass_cutoff":   suggest("highpass_cutoff"),
        "lowpass_cutoff":    suggest("lowpass_cutoff"),
        "target_lufs":       suggest("target_lufs"),
        "eq_gain_db":        suggest("eq_gain_db"),
        "comp_threshold_db": suggest("comp_threshold_db"),
        "comp_ratio":        suggest("comp_ratio"),

        # Denoising
        "denoise":                   OPTIMIZE_DENOISE,
        "noise_reduce_strength":     suggest("noise_reduce_strength") if OPTIMIZE_DENOISE else 0.75,
        "gate_threshold_db":         suggest("gate_threshold_db")     if OPTIMIZE_DENOISE else -30.0,
        "denoise_comp_threshold_db": suggest("denoise_comp_threshold_db") if OPTIMIZE_DENOISE else -16.0,
        "denoise_comp_ratio":        suggest("denoise_comp_ratio")    if OPTIMIZE_DENOISE else 4.0,
        "low_shelf_gain_db":         suggest("low_shelf_gain_db")     if OPTIMIZE_DENOISE else 10.0,
        "denoise_gain_db":           suggest("denoise_gain_db")       if OPTIMIZE_DENOISE else 2.0,
    }

    der = evaluate(params)
    print(f"Trial {trial.number:3d} | DER={der:.4f} | {params}")
    return der


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"{STUDY_NAME}_{timestamp}"

    print(f"Starting optimization — {N_EVAL_FILES} files, {N_TRIALS} trials")
    print(f"Chunks: {_latest}")
    print(f"Audio: {AUDIO_DIR}")
    print(f"Device: {model._diarization.device}")
    print(f"Optimize denoise: {OPTIMIZE_DENOISE}")
    print(f"Study: {study_name}\n")

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=STORAGE,
        load_if_exists=False,
    )
    study.optimize(objective, n_trials=N_TRIALS)

    print("\nBest DER:   ", study.best_value)
    print("Best params:", study.best_params)

    result = {"best_der": study.best_value, "best_params": study.best_params}
    output_path = OUTPUT_DIR / f"best_params_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {output_path}")
