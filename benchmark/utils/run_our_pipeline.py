import os
import sys
import importlib
import torch
import numpy as np

# Make project root importable (preprocess.py lives there)
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

_diarization_pipeline = None
_whisper_model = None


def _load_models():
    global _diarization_pipeline, _whisper_model
    if _diarization_pipeline is not None:
        return

    from dotenv import load_dotenv
    from pyannote.audio import Pipeline
    from transformers import pipeline as asr_pipeline

    load_dotenv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[OurPipeline] Loading diarization pipeline on {device}...")
    hf_token = os.getenv("HF_TOKEN")
    try:
        _diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        )
    except TypeError:
        _diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
    _diarization_pipeline.to(device)

    print("[OurPipeline] Loading Slovenian Whisper model...")
    _whisper_model = asr_pipeline(
        "automatic-speech-recognition",
        model="samolego/whisper-small-slovenian",
        device=0 if device.type == "cuda" else -1,
        chunk_length_s=20,
        stride_length_s=2,
        generate_kwargs={
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.1,
            "language": "sl",
            "task": "transcribe",
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
    from preprocess import load_audio, preprocess, PARAMS

    _load_models()

    t0 = time.time()
    waveform_tensor, sample_rate = load_audio(audio_path)
    waveform_tensor, sample_rate = preprocess(waveform_tensor, sample_rate, PARAMS)
    waveform = waveform_tensor.numpy()[0]
    t_preprocess = time.time() - t0

    t0 = time.time()
    audio_input = {"waveform": waveform_tensor, "sample_rate": sample_rate}
    diarization = _diarization_pipeline(audio_input)
    annotation = diarization.speaker_diarization if hasattr(diarization, "speaker_diarization") else diarization
    raw_segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        raw_segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    raw_segments = _merge_segments(raw_segments)
    t_diarization = time.time() - t0

    t0 = time.time()
    output = []
    for seg in raw_segments:
        start_sample = int(seg["start"] * sample_rate)
        end_sample   = int(seg["end"]   * sample_rate)
        audio_chunk  = waveform[start_sample:end_sample].astype(np.float32)

        duration = seg["end"] - seg["start"]
        if duration < 0.5:
            text = ""
        else:
            result = _whisper_model({"array": audio_chunk, "sampling_rate": sample_rate})
            text = result["text"].strip()

        output.append({
            "speaker": seg["speaker"],
            "start":   round(seg["start"], 3),
            "end":     round(seg["end"],   3),
            "text":    text,
        })
    t_asr = time.time() - t0

    audio_duration = len(waveform) / sample_rate
    print(f"  [OurPipeline] preprocess={t_preprocess:.1f}s  diarization={t_diarization:.1f}s  "
          f"asr={t_asr:.1f}s  total={t_preprocess+t_diarization+t_asr:.1f}s  "
          f"(audio={audio_duration:.1f}s  RTF={( t_preprocess+t_diarization+t_asr)/audio_duration:.2f}x)")

    return sorted(output, key=lambda x: x["start"])
