import os
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.database.util import load_rttm
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HF_TOKEN")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=token
)
pipeline.to(torch.device("cuda"))

diarizationErrorRate = DiarizationErrorRate()

audio_dir = r"datasets\benchmark\VoxConverse\audio"
rttm_dir = r"datasets\benchmark\VoxConverse\rttm"
result_file = r"datasets\benchmark\VoxConverse\result.txt"

der_scores = []

with open(result_file, "w", buffering=1) as result:
    for audio_file in os.listdir(audio_dir):
        if not audio_file.endswith(".wav"):
            continue

        file_name = os.path.splitext(audio_file)[0]
        audio_path = os.path.join(audio_dir, audio_file)
        rttm_path = os.path.join(rttm_dir, file_name + ".rttm")

        if not os.path.exists(rttm_path):
            print(f"No reference RTTM found for {file_name}, skipping...")
            continue

        print(f"Processing {file_name}...")

        # Run diarization
        with ProgressHook() as hook:
            output = pipeline(audio_path, hook=hook)

        hypothesis = output.speaker_diarization

        # Load reference RTTM
        reference = load_rttm(rttm_path)[file_name]

        # Compute DER
        der = diarizationErrorRate(reference, hypothesis)
        der_scores.append(der)

        line = f"{file_name}: DER = {der:.3f}\n"
        print(line, end="")
        result.write(line)
        result.flush()  # write to disk immediately

    # Write average DER
    if der_scores:
        avg_der = sum(der_scores) / len(der_scores)
        avg_line = f"\nAverage DER = {avg_der:.3f}\n"
        print(avg_line, end="")
        result.write(avg_line)
        result.flush()

print("Done! Results saved to", result_file)