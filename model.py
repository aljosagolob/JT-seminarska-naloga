import warnings
warnings.filterwarnings("ignore")

import os
import torch
from pathlib import Path
from pyannote.audio import Pipeline
from dotenv import load_dotenv

from utils import load_audio
from preprocessor import Preprocessor
from transcriber import Transcriber

load_dotenv()


class DiarizationModel:
    def __init__(
        self,
        asr: str | None = "whisper",
        whisper_size: str = "small",
        preprocess_params: dict | None = None,
        pyannote_model: str = "pyannote/speaker-diarization-community-1",
    ):
        self.preprocessor = Preprocessor(preprocess_params)
        self.transcriber = Transcriber(asr, whisper_size) if asr is not None else None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading diarization pipeline ({pyannote_model})...")
        self._diarization = Pipeline.from_pretrained(
            pyannote_model,
            token=os.getenv("HF_TOKEN"),
        )
        self._diarization.to(device)
        print("Diarization pipeline loaded.")

    def set_params(self, params: dict) -> None:
        self.preprocessor.params = params

    def _extract_segments(self, diarization) -> list:
        return [
            {"start": turn.start, "end": turn.end, "speaker": speaker}
            for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True)
        ]

    def _save(self, lines: list[str], output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"Saved to {output_path}")

    def diarize(self, audio_file: str, preprocess: bool = True):
        waveform, sample_rate = load_audio(audio_file)
        if preprocess:
            waveform, sample_rate = self.preprocessor.process(waveform, sample_rate)
        return self._diarization({"waveform": waveform, "sample_rate": sample_rate})

    def execPipeline(self, audio_file: str, output_path: str = "output/transcript.txt") -> list[str]:
        if self.transcriber is None:
            raise RuntimeError("execPipeline requires an ASR model. Recreate DiarizationModel with asr='whisper' or asr='slovenian'.")

        waveform, sample_rate = load_audio(audio_file)
        waveform, sample_rate = self.preprocessor.process(waveform, sample_rate)
        diarization = self._diarization({"waveform": waveform, "sample_rate": sample_rate})

        segments = self._extract_segments(diarization)
        waveform_np = waveform.numpy()[0]

        print("\n=== ANNOTATED TRANSCRIPT ===")
        lines = self.transcriber.run(waveform_np, sample_rate, segments)

        self._save(lines, output_path)
        return lines
