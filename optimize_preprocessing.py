import warnings
warnings.filterwarnings("ignore")

import json
import os
from pathlib import Path
import optuna
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate
import torch
from preprocess import load_audio, preprocess

load_dotenv()

AUDIO_DIR = Path("data/audio_test/audio")
RTTM_DIR = Path("data/transcripts")

# How many files to use per trial -> None uses all
N_EVAL_FILES = 10

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=os.getenv("HF_TOKEN"),
)
pipeline.to(device) # type: ignore
# Runs the diarization model with the prepcrocessing parameters
def evaluate(params: dict) -> float:
    der_metric = DiarizationErrorRate()
    # Get a list of audio files, limited with N_EVAL_FILES
    audio_files = sorted(f for f in AUDIO_DIR.glob("*.wav") if not f.name.startswith("."))[:N_EVAL_FILES]

    for i, audio_path in enumerate(audio_files):
        uri = audio_path.stem
        rttm_path = RTTM_DIR / f"{uri}.rttm"

        print(f"  [{i+1}/{len(audio_files)}] {uri}", end="", flush=True)

        # Load, preprocess, diarize and evaluate
        waveform, sample_rate = load_audio(str(audio_path))
        waveform, sample_rate = preprocess(waveform, sample_rate, params)
        output = pipeline({"waveform": waveform, "sample_rate": sample_rate}) # type: ignore
        reference = load_rttm(str(rttm_path))[uri]
        der_metric(reference, output.speaker_diarization, uem=None)

        print(f" DER={float(abs(der_metric)):.4f}")
    return float(abs(der_metric))

# Called by optuna for each trial, it tries to minimize the returned number: der
def objective(trial: optuna.Trial) -> float:

    # Define the search space for each parameter -> optuna tracks them across trials
    params = {
        "speed":              trial.suggest_float("speed", 0.5, 2.0),
        "highpass_cutoff":    trial.suggest_float("highpass_cutoff", 0.0, 300.0),
        "lowpass_cutoff":     trial.suggest_float("lowpass_cutoff", 4000.0, 8000.0),
        "gain_db":            trial.suggest_float("gain_db", -6.0, 6.0),
        "target_lufs":        trial.suggest_float("target_lufs", -32.0, -16.0),
        "eq_center_freq":     trial.suggest_float("eq_center_freq", 500.0, 4000.0),
        "eq_gain_db":         trial.suggest_float("eq_gain_db", -6.0, 12.0),
        "eq_q":               trial.suggest_float("eq_q", 0.5, 3.0),
        "comp_threshold_db":  trial.suggest_float("comp_threshold_db", -40.0, -10.0),
        "comp_ratio":         trial.suggest_float("comp_ratio", 1.0, 8.0),
    }
    der = evaluate(params)
    print(f"Trial {trial.number:3d} | DER={der:.4f} | {params}")
    return der


if __name__ == "__main__":
    print(f"Starting optimization on {N_EVAL_FILES} files, 50 trials...")
    print(f"Audio: {AUDIO_DIR}")
    print(f"Device: {device}\n")

    # Create optuna instance
    study = optuna.create_study(
        direction="minimize",
        study_name="diarization_preprocessing",
        storage="sqlite:///optuna.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=50)

    print("\nBest DER:", study.best_value)
    print("Best params:", study.best_params)

    result = {"best_der": study.best_value, "best_params": study.best_params}
    with open("best_params.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Saved to best_params.json")