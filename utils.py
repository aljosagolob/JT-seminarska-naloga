import soundfile as sf
import torch


def load_audio(path: str) -> tuple[torch.Tensor, int]:
    audio, sample_rate = sf.read(path, dtype="float32")
    waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, time)
    return waveform, sample_rate
