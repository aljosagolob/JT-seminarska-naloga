import warnings
warnings.filterwarnings("ignore")

import torch
import soundfile as sf
from pyannote.audio import Pipeline
from dotenv import load_dotenv
from transformers import pipeline as asr_pipeline
import numpy as np
import os

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


load_dotenv()

#   D I A R I Z A T I O N
print("Loading diarization pipeline...")
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=os.getenv("HF_TOKEN"),
)
print("Pipeline loaded...")

#   M O V E   T O   G P U   I F   A V A I L A B L E
if torch.cuda.is_available():
    print("Running on cuda")
    diarization_pipeline.to(torch.device("cuda"))

#   L O A D   A U D I O
audio_file = "datasets/artur-j/wav/Artur-J-Gvecg-P500030-avd.wav"
waveform, sample_rate = sf.read(audio_file)
if waveform.ndim > 1:
    waveform = waveform.mean(axis=1)
waveform_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
audio_input = {"waveform": waveform_tensor, "sample_rate": sample_rate}

print("Running diarization...")
diarization = diarization_pipeline(audio_input)

#   S A V E   S E G M E N T S   I N   L I S T
segments = []
annotation = diarization.speaker_diarization if hasattr(diarization, "speaker_diarization") else diarization
for turn, _, speaker in annotation.itertracks(yield_label=True):
    # print(f"[{turn.start} - {turn.end}] - {speaker}")
    segments.append({
        "start": turn.start,
        "end": turn.end,
        "speaker": speaker
    })

#   M E R G E   S E G M E N T S
segments = merge_segments(segments=segments)

#   L O A D   S L O V E E N E   W H I S P E R   M O D E L
asr = asr_pipeline(
    "automatic-speech-recognition",
    model="samolego/whisper-small-slovenian",
    device=0 if torch.cuda.is_available() else -1,
)

#   C O M B I N E
print("\n=== ANNOTATED TRANSCRIPT ===")
output_lines = []

for seg in segments:
    start_sample = int(seg["start"] * sample_rate)
    end_sample = int(seg["end"] * sample_rate)
    
    audio_chunk  = waveform[start_sample:end_sample].astype(np.float32)
    
    # Skips segment if less than 0.5s
    duration = seg["end"] - seg["start"]
    if duration < 0.5: 
        line = f'{seg["speaker"]} [{seg["start"]:.2f}s-{seg["end"]:.2f}s]: [prekratko]'
    else:
        result = asr({"array": audio_chunk, "sampling_rate": sample_rate})
        text = result["text"].strip()
        line = f'{seg["speaker"]} [{seg["start"]:.2f}s-{seg["end"]:.2f}s]: {text}'
    
    print(line)
    output_lines.append(line)

#   S A V E   R E S U L T
with open("output/transcript.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))

print("Saved to output/transcript.txt") 
