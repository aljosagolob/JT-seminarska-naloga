import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import sys
import torch
import torchaudio
import numpy as np
import soundfile as sf
from dotenv import load_dotenv

_torch_load_orig = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs["weights_only"] = False
    return _torch_load_orig(*args, **kwargs)
torch.load = _torch_load_compat

if not hasattr(torchaudio, "AudioMetaData"):
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
        import soundfile as _sf
        info = _sf.info(path)
        return torchaudio.AudioMetaData(
            sample_rate=info.samplerate, num_frames=info.frames,
            num_channels=info.channels, bits_per_sample=16, encoding="PCM_S",
        )
    torchaudio.info = _torchaudio_info

from pyannote.audio import Pipeline
from preprocessor import preprocess, PARAMS
from utils import load_audio


def merge_segments(segments, gap_threshold=0.5):
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


def load_asr_model(model_type: str, device):
    if model_type == "sl":
        from transformers import pipeline as asr_pipeline
        print("Loading Slovenian Whisper model...")
        asr = asr_pipeline(
            "automatic-speech-recognition",
            model="samolego/whisper-small-slovenian",
            device=0 if device.type == "cuda" else -1,
        )
        return ("hf", asr)
    else:
        import whisper
        print("Loading Whisper model...")
        model = whisper.load_model("small")
        if device.type == "cuda":
            model = model.to(device)
        return ("whisper", model)


def transcribe_chunk(asr_model, audio_chunk: np.ndarray, sample_rate: int) -> str:
    kind, model = asr_model
    if kind == "hf":
        result = model({"array": audio_chunk, "sampling_rate": sample_rate})
        return result["text"].strip()
    else:
        result = model.transcribe(audio_chunk, language="sl")
        return result["text"].strip()


def run(audio_file: str, model_type: str, use_preprocess: bool, output_path: str | None):
    if not os.path.exists(audio_file):
        print(f"Error: audio file not found: {audio_file}", file=sys.stderr)
        sys.exit(1)

    load_dotenv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading diarization pipeline...")
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.getenv("HF_TOKEN"),
    )
    diarization_pipeline.to(device)

    if use_preprocess:
        print("Loading and preprocessing audio...")
        waveform_tensor, sample_rate = load_audio(audio_file)
        waveform_tensor, sample_rate = preprocess(waveform_tensor, sample_rate, PARAMS)
        waveform = waveform_tensor.numpy()[0]
    else:
        print("Loading audio...")
        waveform, sample_rate = sf.read(audio_file)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        waveform_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)

    audio_input = {"waveform": waveform_tensor, "sample_rate": sample_rate}

    print("Running diarization...")
    diarization = diarization_pipeline(audio_input)

    segments = []
    for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
        segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    segments = merge_segments(segments)

    asr_model = load_asr_model(model_type, device)

    print("\n=== ANNOTATED TRANSCRIPT ===")
    output_lines = []

    for seg in segments:
        start_sample = int(seg["start"] * sample_rate)
        end_sample = int(seg["end"] * sample_rate)
        audio_chunk = waveform[start_sample:end_sample].astype(np.float32)

        duration = seg["end"] - seg["start"]
        if duration < 0.5:
            text = "[too short]"
        else:
            text = transcribe_chunk(asr_model, audio_chunk, sample_rate)

        line = f'{seg["speaker"]} [{seg["start"]:.2f}s-{seg["end"]:.2f}s]: {text}'
        print(line)
        output_lines.append(line)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))
        print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run speaker diarization + transcription on an audio file.")
    parser.add_argument("audio", help="Path to the input .wav audio file")
    parser.add_argument(
        "--model", choices=["whisper", "sl"], default="whisper",
        help="ASR model: 'whisper' (multilingual Whisper small) or 'sl' (Slovenian fine-tuned). Default: whisper"
    )
    parser.add_argument(
        "--preprocess", action="store_true",
        help="Apply audio preprocessing before diarization"
    )
    parser.add_argument(
        "--output", default="output/transcript.txt",
        help="Path to save the transcript. Default: output/transcript.txt"
    )
    args = parser.parse_args()

    run(args.audio, args.model, args.preprocess, args.output)
