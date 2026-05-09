import warnings
warnings.filterwarnings("ignore")

import torch
import soundfile as sf
from pyannote.audio import Pipeline
import whisper

#   D I A R I Z A T I O N
print("Loading diarization pipeline...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=os.getenv("HF_TOKEN"),
)
diarization_pipeline.to(device) # type: ignore
print("Pipeline loaded...")

#   L O A D   A U D I O
audio_file = "data/convo.wav"
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
    segments.append({
        "start": turn.start,
        "end": turn.end,
        "speaker": speaker
    })

#   T R A N S C R I P T I O N
print("Loading Whisper...")
whisper_model = whisper.load_model("base") 

print("Running transcription...")
result = whisper_model.transcribe(audio_file)
whisper_segments = result["segments"]

#   C O M B I N E
print("\n=== ANNOTATED TRANSCRIPT ===")
output_lines = []

for seg in segments:
    text_parts = []
    for w_seg in whisper_segments:
        overlap = min(seg["end"], w_seg["end"]) - max(seg["start"], w_seg["start"])
        if overlap > 0:
            text_parts.append(w_seg["text"].strip())
    
    text = " ".join(text_parts) if text_parts else "[tišina]"
    line = f'{seg["speaker"]}: {text}'
    print(line)
    output_lines.append(line)

#   S A V E   R E S U L T
with open("output/transcript.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))

print("\nShranjeno v output/transcript.txt")
