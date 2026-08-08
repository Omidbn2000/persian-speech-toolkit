import soundfile as sf
import torch
import os
from silero_vad import load_silero_vad, get_speech_timestamps

# Load everything
audio, sr = sf.read(r'audio_sample\tozih.opus')
wav = torch.from_numpy(audio if len(audio.shape) == 1 else audio.mean(axis=1)).float()
model = load_silero_vad()
segments = get_speech_timestamps(wav, model, return_seconds=True)

# Create output folder
os.makedirs("segments", exist_ok=True)

# Save each segment
for i, seg in enumerate(segments, 1):
    start = int(seg['start'] * sr)
    end = int(seg['end'] * sr)
    sf.write(f"segments/segment_{i}.wav", audio[start:end], sr)
    print(f"Saved segment {i}: {seg['start']:.2f}s - {seg['end']:.2f}s")

print(f"\n✅ Saved {len(segments)} segments to 'segments' folder")