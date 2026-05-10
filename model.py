import warnings
warnings.filterwarnings("ignore")

import torch
import soundfile as sf
from pyannote.audio import Pipeline
from dotenv import load_dotenv
import whisper
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

#   L O A D   D I A R I Z A T I O N   P I P E L I N E
print("Loading diarization pipeline...")
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=os.getenv("HF_TOKEN"),
)
print("Pipeline loaded...")

#   M O V E   T O   G P U   I F   A V A I L A B L E
if torch.cuda.is_available():
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
for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
    # print(f"[{turn.start} - {turn.end}] - {speaker}")
    segments.append({
        "start": turn.start,
        "end": turn.end,
        "speaker": speaker
    })

#   M E R G E   S E G M E N T S
segments = merge_segments(segments=segments)

#   L O A D   W H I S P E R   M O D E L
print("Loading Whisper...")
whisper_model = whisper.load_model("small") 

#   M O V E   T O   G P U   I F   A V A I L A B L E
if torch.cuda.is_available():
    whisper_model = whisper_model.to(torch.device("cuda"))

#   C O M B I N E
print("\n=== ANNOTATED TRANSCRIPT ===")
output_lines = []

for seg in segments:
    start_sample = int(seg["start"] * sample_rate)
    end_sample = int(seg["end"] * sample_rate)
    
    audio_chunk = waveform[start_sample:end_sample].astype(np.float32)
    
    # Skips segment if less than 0.5s
    duration = seg["end"] - seg["start"]
    if duration < 0.5: 
        line = f'{seg["speaker"]} [{seg["start"]:.2f}s-{seg["end"]:.2f}s]: [too short]'
    else:
        result = whisper_model.transcribe(audio_chunk, language="sl")
        text = result["text"].strip()
        line = f'{seg["speaker"]} [{seg["start"]:.2f}s-{seg["end"]:.2f}s]: {text}'
    
    print(line)
    output_lines.append(line)

#   S A V E   R E S U L T
with open("output/transcript.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))

print("Saved to output/transcript.txt") 
