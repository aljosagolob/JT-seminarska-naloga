import torch
import torchaudio.functional as F
from effects import bandpass_eq


PARAMS = {
    # --- Acoustic preprocessing ---
    "speed":              1.0,
    "highpass_cutoff":    59.42,
    "lowpass_cutoff":     6193.48,
    "target_lufs":       -27.43,
    "eq_gain_db":         -2.08,
    "comp_threshold_db": -13.43,
    "comp_ratio":         3.79,

    # --- Denoising ---
    "denoise":                   True,
    "noise_reduce_strength":      0.22,
    "gate_threshold_db":         -46.69,
    "denoise_comp_threshold_db": -13.64,
    "denoise_comp_ratio":          7.14,
    "low_shelf_gain_db":           4.10,
    "denoise_gain_db":            -1.78,
}


def preprocess(waveform: torch.Tensor, sample_rate: int, params: dict = PARAMS) -> tuple[torch.Tensor, int]:
    if params.get("denoise", False):
        waveform, sample_rate = noise_reduction(waveform, sample_rate, params)

    if params["speed"] != 1.0:
        waveform = F.resample(waveform, orig_freq=int(sample_rate * params["speed"]), new_freq=sample_rate)

    if params["highpass_cutoff"] > 0 or params["lowpass_cutoff"] < sample_rate / 2:
        audio_np = waveform.numpy()[0]
        audio_np = bandpass_eq(audio_np, sample_rate, lo_hz=params["highpass_cutoff"], hi_hz=params["lowpass_cutoff"])
        waveform = torch.tensor(audio_np).unsqueeze(0)

    rms = waveform.pow(2).mean().sqrt()
    if rms > 0:
        waveform = waveform * (10 ** (params["target_lufs"] / 20) / rms)

    if params["eq_gain_db"] != 0.0:
        waveform = F.equalizer_biquad(
            waveform, sample_rate,
            center_freq=2000.0,
            gain=params["eq_gain_db"],
            Q=1.0,
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

    return waveform, sample_rate


def noise_reduction(waveform: torch.Tensor, sample_rate: int, params: dict) -> tuple[torch.Tensor, int]:
    import noisereduce as nr
    from pedalboard._pedalboard import Pedalboard
    from pedalboard import NoiseGate, Compressor, LowShelfFilter, Gain

    audio_np = waveform.numpy()[0]

    cleaned = nr.reduce_noise(
        y=audio_np, sr=sample_rate,
        stationary=True,
        prop_decrease=params["noise_reduce_strength"],
    )

    board = Pedalboard([
        NoiseGate(threshold_db=params["gate_threshold_db"], ratio=1.5, release_ms=250),
        Compressor(threshold_db=params["denoise_comp_threshold_db"], ratio=params["denoise_comp_ratio"]),
        LowShelfFilter(cutoff_frequency_hz=400, gain_db=params["low_shelf_gain_db"], q=1),
        Gain(gain_db=params["denoise_gain_db"]),
    ])

    enhanced = board(cleaned.reshape(1, -1), sample_rate).flatten()
    return torch.tensor(enhanced).unsqueeze(0), sample_rate


class Preprocessor:
    def __init__(self, params: dict | None = None):
        self.params = params or PARAMS.copy()

    def process(self, waveform: torch.Tensor, sample_rate: int) -> tuple[torch.Tensor, int]:
        return preprocess(waveform, sample_rate, self.params)
