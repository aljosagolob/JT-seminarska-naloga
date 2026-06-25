"""
download_ami.py
Downloads 5 AMI test files via HuggingFace streaming.
No manual URLs, no 404s.

pip install datasets soundfile numpy
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from datasets import load_dataset
import os

os.environ["HF_HOME"] = "D:\\huggingface_cache"

AUDIO_DIR = Path("data/ami/test/audio")
RTTM_DIR = Path("data/ami/test/rttm")

N_FILES = 5


if __name__ == "__main__":
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    RTTM_DIR.mkdir(parents=True, exist_ok=True)

    print("Connecting to HuggingFace (streaming, no full download)...")
    ds = load_dataset(
        "diarizers-community/ami",
        "ihm",
        split="test",
        streaming=True,
        trust_remote_code=True,
    )

    for i, sample in enumerate(ds):
        if i >= N_FILES:
            break

        uri = f"meeting_{i:03d}"
        print(f"\n[{i+1}/{N_FILES}] Saving {uri}...")

        # Save audio
        audio_array = np.array(sample["audio"]["array"])
        sample_rate = sample["audio"]["sampling_rate"]
        audio_path = AUDIO_DIR / f"{uri}.wav"
        sf.write(str(audio_path), audio_array, sample_rate)
        print(f"  Audio: {audio_path}  ({len(audio_array)/sample_rate:.1f}s)")

        # Save RTTM
        rttm_path = RTTM_DIR / f"{uri}.rttm"
        with open(rttm_path, "w") as f:
            starts = sample["timestamps_start"]
            ends = sample["timestamps_end"]
            speakers = sample["speakers"]
            for start, end, speaker in zip(starts, ends, speakers):
                duration = end - start
                f.write(
                    f"SPEAKER {uri} 1 {start:.3f} {duration:.3f} "
                    f"<NA> <NA> {speaker} <NA> <NA>\n"
                )
        print(f"  RTTM: {rttm_path}  ({len(starts)} segments)")

    print(f"\nDone.")
    print(f"Audio files : {sorted(AUDIO_DIR.glob('*.wav'))}")
    print(f"RTTM files  : {sorted(RTTM_DIR.glob('*.rttm'))}")
