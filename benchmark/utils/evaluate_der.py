from pyannote.metrics.diarization import DiarizationErrorRate
from .rttm_io import load_rttm, segments_to_annotation


def evaluate_der(gold_rttm_path: str, hypothesis_segments: list, collar: float = 0.0) -> dict:
    metric = DiarizationErrorRate(collar=collar, skip_overlap=False)

    ref = load_rttm(gold_rttm_path)
    hyp = segments_to_annotation(hypothesis_segments)

    detail = metric(ref, hyp, detailed=True)
    total = detail["total"]

    if total == 0:
        return {"der": 0.0, "miss": 0.0, "fa": 0.0, "confusion": 0.0, "total_duration": 0.0}

    return {
        "der":            detail["diarization error rate"],
        "miss":           detail["missed detection"] / total,
        "fa":             detail["false alarm"] / total,
        "confusion":      detail["confusion"] / total,
        "total_duration": total,
    }
