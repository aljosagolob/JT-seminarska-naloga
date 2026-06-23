import torch
import torchaudio.functional as F
import noisereduce as nr

PARAMS = {
    "speed":              1.0,
    "highpass_cutoff":    0.0,
    "lowpass_cutoff":     8000.0,
    "gain_db":            0.0,
    "target_lufs":       -23.0,
    "eq_center_freq":    2000.0,
    "eq_gain_db":         0.0,
    "eq_q":               1.0,
    "comp_threshold_db": -20.0,
    "comp_ratio":         1.0,
    "noise_reduce":       0.0,
}


def preprocess(waveform: torch.Tensor, sample_rate: int, params: dict = PARAMS) -> tuple[torch.Tensor, int]:
    if params["speed"] != 1.0:
        waveform = F.resample(waveform, orig_freq=int(sample_rate * params["speed"]), new_freq=sample_rate)

    if params["highpass_cutoff"] > 0:
        waveform = F.highpass_biquad(waveform, sample_rate, cutoff_freq=params["highpass_cutoff"])

    if params["lowpass_cutoff"] < sample_rate / 2:
        waveform = F.lowpass_biquad(waveform, sample_rate, cutoff_freq=params["lowpass_cutoff"])

    rms = waveform.pow(2).mean().sqrt()
    if rms > 0:
        waveform = waveform * (10 ** (params["target_lufs"] / 20) / rms)

    if params["eq_gain_db"] != 0.0:
        waveform = F.equalizer_biquad(
            waveform, sample_rate,
            center_freq=params["eq_center_freq"],
            gain=params["eq_gain_db"],
            Q=params["eq_q"],
        )

    if params["comp_ratio"] > 1.0:
        amplitude = waveform.abs().clamp(min=1e-8)
        db = 20 * torch.log10(amplitude)
        gain_db = torch.where(
            db > params["comp_threshold_db"],
            (params["comp_threshold_db"] - db) * (1.0 - 1.0 / params["comp_ratio"]),
            torch.zeros_like(db),
        )
        waveform = waveform * (10 ** (gain_db / 20))

    if params["gain_db"] != 0.0:
        waveform = F.gain(waveform, params["gain_db"])

    if params.get("noise_reduce", 0.0) > 0.0:
        audio_np = waveform.numpy()[0]
        reduced = nr.reduce_noise(y=audio_np, sr=sample_rate, prop_decrease=params["noise_reduce"])
        waveform = torch.tensor(reduced).unsqueeze(0)

    return waveform, sample_rate


class Preprocessor:
    def __init__(self, params: dict | None = None):
        self.params = params or PARAMS.copy()

    def process(self, waveform: torch.Tensor, sample_rate: int) -> tuple[torch.Tensor, int]:
        return preprocess(waveform, sample_rate, self.params)
