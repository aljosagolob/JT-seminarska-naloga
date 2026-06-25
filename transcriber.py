import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np


class Transcriber:
    def __init__(self, asr: str = "whisper", whisper_size: str = "small"):
        if asr not in ("whisper", "slovenian"):
            raise ValueError(f"asr must be 'whisper' or 'slovenian', got '{asr}'")

        self.asr_mode = asr
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading ASR model...")
        if asr == "whisper":
            import whisper
            self._model = whisper.load_model(whisper_size, device=device)
            self._pipeline = None
        else:
            from transformers import pipeline as asr_pipeline
            self._model = None
            self._pipeline = asr_pipeline(
                "automatic-speech-recognition",
                model="samolego/whisper-small-slovenian",
                device=0 if torch.cuda.is_available() else -1,
            )
        print("ASR model loaded.")

    def _merge_segments(self, segments: list, gap_threshold: float = 0.5) -> list:
        if not segments:
            return segments
        merged = [segments[0].copy()]
        for current in segments[1:]:
            previous = merged[-1]
            if current["speaker"] == previous["speaker"] and (current["start"] - previous["end"]) < gap_threshold:
                previous["end"] = current["end"]
            else:
                merged.append(current.copy())
        return merged

    def run(self, waveform_np: np.ndarray, sample_rate: int, segments: list) -> list[str]:
        segments = self._merge_segments(segments)
        lines = []
        for seg in segments:
            start_sample = int(seg["start"] * sample_rate)
            end_sample = int(seg["end"] * sample_rate)
            chunk = waveform_np[start_sample:end_sample].astype(np.float32)
            duration = seg["end"] - seg["start"]

            if duration < 0.5:
                text = "[too short]"
            elif self.asr_mode == "whisper":
                result = self._model.transcribe(chunk, language="sl")
                text = result["text"].strip()
            else:
                result = self._pipeline({"array": chunk, "sampling_rate": sample_rate})
                text = result["text"].strip()

            line = f'{seg["speaker"]} [{seg["start"]:.2f}s-{seg["end"]:.2f}s]: {text}'
            print(line)
            lines.append(line)

        return lines
