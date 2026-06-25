import os
import sys
import importlib
import torch
import numpy as np

# Make project root importable (model.py lives there)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_torch_load_orig = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs["weights_only"] = False
    return _torch_load_orig(*args, **kwargs)
torch.load = _torch_load_compat

# Patch torchaudio API removed in 2.x — must happen before pyannote imports
import torchaudio
if not hasattr(torchaudio, "AudioMetaData"):
    _found = False
    for _path in ("torchaudio.backend.common", "torchaudio._backend.utils",
                  "torchaudio._backend.common", "torchaudio.io", "torchaudio.io._compat"):
        try:
            _m = importlib.import_module(_path)
            if hasattr(_m, "AudioMetaData"):
                torchaudio.AudioMetaData = _m.AudioMetaData
                _found = True
                break
        except (ImportError, ModuleNotFoundError):
            continue
    if not _found:
        from dataclasses import dataclass
        @dataclass
        class _AudioMetaData:
            sample_rate: int = 0
            num_frames: int = 0
            num_channels: int = 0
            bits_per_sample: int = 0
            encoding: str = ""
        torchaudio.AudioMetaData = _AudioMetaData
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
if not hasattr(torchaudio, "get_audio_backend"):
    torchaudio.get_audio_backend = lambda: "soundfile"
if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda backend: None
if not hasattr(torchaudio, "info"):
    def _torchaudio_info(path, format=None):
        import soundfile as sf
        info = sf.info(path)
        return torchaudio.AudioMetaData(
            sample_rate=info.samplerate, num_frames=info.frames,
            num_channels=info.channels, bits_per_sample=16, encoding="PCM_S",
        )
    torchaudio.info = _torchaudio_info

_model = None
_whisper_pipeline = None


def _load_models():
    global _model, _whisper_pipeline
    if _model is not None:
        return

    from model import DiarizationModel
    from transformers import pipeline as asr_pipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[OurPipeline] Loading DiarizationModel on {device}...")
    _model = DiarizationModel(
        asr=None,
        pyannote_model="pyannote/speaker-diarization-3.1",
    )

    print("[OurPipeline] Loading Slovenian Whisper model...")
    _whisper_pipeline = asr_pipeline(
        "automatic-speech-recognition",
        model="shripadbhat/whisper-medium-sl",
        device=0 if device.type == "cuda" else -1,
        generate_kwargs={
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.1,
        }
    )

    print("[OurPipeline] Models ready.")


def _merge_segments(segments: list, gap_threshold: float = 0.5) -> list:
    if not segments:
        return segments
    merged = [segments[0].copy()]
    for current in segments[1:]:
        previous = merged[-1]
        gap = current["start"] - previous["end"]
        if current["speaker"] == previous["speaker"] and gap < gap_threshold:
            previous["end"] = current["end"]
        else:
            merged.append(current.copy())
    return merged


def run_our_pipeline(audio_path: str) -> list:
    """
    Run our pipeline on a single WAV file.
    Returns: [{"speaker": str, "start": float, "end": float, "text": str}, ...]
    """
    import time
    import soundfile as sf

    _load_models()

    t0 = time.time()
    raw_audio, sr = sf.read(audio_path, dtype="float32")
    if raw_audio.ndim > 1:
        raw_audio = raw_audio.mean(axis=1)
    diarization = _model.diarize(audio_path)
    t_preprocess_diarize = time.time() - t0

    annotation = diarization.speaker_diarization if hasattr(diarization, "speaker_diarization") else diarization
    raw_segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        raw_segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    raw_segments = _merge_segments(raw_segments)

    t0 = time.time()
    output = []
    for seg in raw_segments:
        start_sample = int(seg["start"] * sr)
        end_sample   = int(seg["end"]   * sr)
        audio_chunk  = raw_audio[start_sample:end_sample].astype(np.float32)

        duration = seg["end"] - seg["start"]
        if duration < 0.5:
            text = ""
        else:
            result = _whisper_pipeline({"array": audio_chunk, "sampling_rate": sr})
            text = result["text"].strip()

        output.append({
            "speaker": seg["speaker"],
            "start":   round(seg["start"], 3),
            "end":     round(seg["end"],   3),
            "text":    text,
        })
    t_asr = time.time() - t0

    audio_duration = len(raw_audio) / sr
    total = t_preprocess_diarize + t_asr
    print(f"  [OurPipeline] preprocess+diarization={t_preprocess_diarize:.1f}s  asr={t_asr:.1f}s  "
          f"total={total:.1f}s  (audio={audio_duration:.1f}s  RTF={total/audio_duration:.2f}x)")

    return sorted(output, key=lambda x: x["start"])
