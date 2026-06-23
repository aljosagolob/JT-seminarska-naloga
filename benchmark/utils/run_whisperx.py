import os
import torch

_whisperx_model = None
_align_model = None
_align_meta = None
_diarize_model = None


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_models():
    global _whisperx_model, _align_model, _align_meta, _diarize_model
    if _whisperx_model is not None:
        return

    import whisperx

    device = _get_device()
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"[WhisperX] Loading models on {device} ({compute_type})...")

    _whisperx_model = whisperx.load_model(
        "large-v3", device, compute_type=compute_type, language="sl"
    )

    try:
        _align_model, _align_meta = whisperx.load_align_model(
            language_code="sl", device=device
        )
        print("[WhisperX] Using Slovenian alignment model")
    except Exception:
        print("[WhisperX] Slovenian alignment model not found, using Croatian (hr)")
        _align_model, _align_meta = whisperx.load_align_model(
            language_code="hr", device=device
        )

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("HF_TOKEN environment variable not set")

    _diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=hf_token, device=device
    )
    print("[WhisperX] Models ready.")


def run_whisperx(audio_path: str) -> list:
    import whisperx

    _load_models()

    audio = whisperx.load_audio(audio_path)

    result = _whisperx_model.transcribe(audio, batch_size=16, language="sl")
    result = whisperx.align(
        result["segments"], _align_model, _align_meta, audio, _get_device(),
        return_char_alignments=False
    )
    diarize_segments = _diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    segments = []
    for seg in result["segments"]:
        segments.append({
            "speaker": seg.get("speaker", "SPEAKER_UNKNOWN"),
            "start":   round(seg["start"], 3),
            "end":     round(seg["end"], 3),
            "text":    seg["text"].strip(),
        })

    return sorted(segments, key=lambda x: x["start"])
