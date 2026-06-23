import re
from jiwer import wer as jiwer_wer


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def evaluate_wer(ref_segments: list, hyp_segments: list) -> dict:
    ref_text = normalize(
        " ".join(s["text"] for s in sorted(ref_segments, key=lambda x: x["start"]))
    )
    hyp_text = normalize(
        " ".join(s["text"] for s in sorted(hyp_segments, key=lambda x: x["start"]))
    )

    if not ref_text:
        return {"wer": None, "ref_words": 0, "hyp_words": len(hyp_text.split())}

    return {
        "wer":       jiwer_wer(ref_text, hyp_text),
        "ref_words": len(ref_text.split()),
        "hyp_words": len(hyp_text.split()),
    }
