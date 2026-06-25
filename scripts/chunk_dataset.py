"""
Splits long audio files into shorter disjoint chunks using RTTM boundaries.
Only keeps chunks that contain at least MIN_SPEAKERS unique speakers.
Splits only at segment boundaries so cuts are always clean.
"""

import sys
from datetime import datetime
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Config ────────────────────────────────────────────────────────────────────
AUDIO_DIR = Path("datasets/artur-j/wav")
RTTM_DIR = Path("datasets/artur-j/rttm")
OUT_BASE = Path("datasets/artur-j/chunks")

TARGET_DURATION = 30.0  # seconds — target max chunk length
MIN_SPEAKERS = 2        # minimum unique speakers required per chunk


def parse_rttm(rttm_path: Path) -> list[dict]:
    segments = []
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] != "SPEAKER":
                continue
            segments.append({
                "uri": parts[1],
                "start": float(parts[3]),
                "duration": float(parts[4]),
                "end": float(parts[3]) + float(parts[4]),
                "speaker": parts[7],
            })
    return sorted(segments, key=lambda s: s["start"])


def write_rttm(segments: list[dict], uri: str, path: Path, offset: float) -> None:
    with open(path, "w") as f:
        for seg in segments:
            start = seg["start"] - offset
            duration = seg["duration"]
            f.write(f"SPEAKER {uri} 1 {start:.3f} {duration:.3f} <NA> <NA> {seg['speaker']} <NA> <NA>\n")


def chunk_file(audio_path: Path, rttm_path: Path, out_audio: Path, out_rttm: Path) -> int:
    segments = parse_rttm(rttm_path)
    if not segments:
        return 0

    audio, sr = sf.read(audio_path)
    stem = audio_path.stem
    chunk_idx = 0
    saved = 0

    i = 0
    while i < len(segments):
        chunk_segs = []
        chunk_start = segments[i]["start"]

        while i < len(segments):
            seg = segments[i]
            chunk_segs.append(seg)
            chunk_end = seg["end"]
            if chunk_end - chunk_start >= TARGET_DURATION:
                i += 1
                break
            i += 1

        speakers = {s["speaker"] for s in chunk_segs}
        if len(speakers) < MIN_SPEAKERS:
            chunk_idx += 1
            continue

        start_sample = int(chunk_start * sr)
        end_sample = int(chunk_end * sr)
        audio_chunk = audio[start_sample:end_sample]

        uri = f"{stem}_chunk_{chunk_idx:03d}"
        out_wav = out_audio / f"{uri}.wav"
        out_rttm_path = out_rttm / f"{uri}.rttm"

        sf.write(out_wav, audio_chunk, sr)
        write_rttm(chunk_segs, uri, out_rttm_path, offset=chunk_start)

        print(f"  {uri}: {chunk_end - chunk_start:.1f}s, {len(speakers)} speakers")
        chunk_idx += 1
        saved += 1

    return saved


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_BASE / timestamp
    out_audio = out_dir / "audio"
    out_rttm = out_dir / "rttm"
    out_audio.mkdir(parents=True, exist_ok=True)
    out_rttm.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(AUDIO_DIR.glob("*.wav"))
    if not audio_files:
        print(f"No WAV files found in {AUDIO_DIR}")
        return

    print(f"Output folder: {out_dir}\n")

    total = 0
    for audio_path in audio_files:
        rttm_path = RTTM_DIR / f"{audio_path.stem}.rttm"
        if not rttm_path.exists():
            print(f"Skipping {audio_path.name} — no matching RTTM")
            continue

        print(f"{audio_path.name}")
        saved = chunk_file(audio_path, rttm_path, out_audio, out_rttm)
        print(f"  → {saved} chunks saved")
        total += saved

    print(f"\nDone. {total} total chunks in {out_dir}")


if __name__ == "__main__":
    main()
