import soundfile as sf
import torch
import torchaudio.functional as F

# Default params for preprocessing
PARAMS = {
    # Basic stuff
    "speed":              1.0,      # Audio file speed
    "highpass_cutoff":    0.0,      # Cuts off frequencies below this
    "lowpass_cutoff":     8000.0,   # Cuts off frequencies above this
    "gain_db":            0.0,      # Loudness boost/decrease in dB
    "target_lufs":       -23.0,     # loudness that all audio should be normalized to

    # EQ / Bell filter -> boosta frekvence okoli sredine bell krivulje
    "eq_center_freq":    2000.0,  # Center frequency for EQ
    "eq_gain_db":         0.0,    # dB boost/cutfor EQ at centre
    "eq_q":               1.0,    # how wide the bell filter is

    # Dynamic range compression
    "comp_threshold_db": -20.0,   # sound above this threshold (too loud) will be compressed
    "comp_ratio":         1.0,    # how strong should the compression be 
}


def load_audio(path: str) -> tuple[torch.Tensor, int]:
    audio, sample_rate = sf.read(path, dtype="float32")
    waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, time)
    return waveform, sample_rate


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

    return waveform, sample_rate
