import numpy as np
from scipy.signal import butter, sosfilt


def bandpass_eq(audio: np.ndarray, sample_rate: int, lo_hz: float = 0.0, hi_hz: float = 8000.0) -> np.ndarray:
    nyq = sample_rate / 2.0
    apply_hi = lo_hz > 0
    apply_lo = hi_hz < nyq

    if apply_hi and apply_lo:
        sos = butter(4, [lo_hz / nyq, hi_hz / nyq], btype="band", output="sos")
    elif apply_hi:
        sos = butter(4, lo_hz / nyq, btype="high", output="sos")
    elif apply_lo:
        sos = butter(4, hi_hz / nyq, btype="low", output="sos")
    else:
        return audio

    return sosfilt(sos, audio)
