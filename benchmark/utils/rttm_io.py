from pyannote.core import Annotation, Segment
import os


def load_rttm(rttm_path: str) -> Annotation:
    ann = Annotation()
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 10 or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            ann[Segment(start, start + duration)] = speaker
    return ann


def segments_to_annotation(segments: list) -> Annotation:
    ann = Annotation()
    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        if end - start <= 0.01:
            continue
        ann[Segment(start, end)] = seg["speaker"]
    return ann


def find_rttm(file_id: str, rttm_dir: str) -> str | None:
    path = os.path.join(rttm_dir, f"{file_id}.rttm")
    return path if os.path.exists(path) else None
