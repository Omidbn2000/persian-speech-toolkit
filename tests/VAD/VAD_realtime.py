import sounddevice as sd
import numpy as np
import torch
from silero_vad import load_silero_vad, get_speech_timestamps
from collections import deque

def realtime_vad_monitor():
    """Real-time Voice Activity Detection on microphone input"""
    
    # Load VAD model
    print("Loading Silero VAD model...")
    model = load_silero_vad()
    print("Model loaded successfully!")
    print("-" * 50)
    
    # Settings
    sample_rate = 16000  # Silero VAD works best with 16kHz
    model_frame_size = 512  # Silero VAD expects 512 samples at 16kHz
    buffer_duration = 1.0  # Keep 1 second of audio for context
    buffer_samples = int(sample_rate * buffer_duration)
    
    # Buffer to store recent audio
    audio_buffer = deque(maxlen=buffer_samples)
    
    # State tracking
    speech_active = False
    speech_frames = 0
    silence_frames = 0
    min_speech_frames = 3  # Need at least 3 consecutive frames to confirm speech
    min_silence_frames = 5  # Need at least 5 consecutive silence frames to confirm silence
    
    def callback(indata, frames, time, status):
        nonlocal speech_active, speech_frames, silence_frames
        
        if status:
            print(f"\nStatus: {status}")
        
        # Convert to mono if needed and add to buffer
        audio_chunk = indata.flatten()
        audio_buffer.extend(audio_chunk)
        
        # Process when we have enough data for one VAD frame
        if len(audio_buffer) >= model_frame_size:
            # Get exactly 512 samples
            audio_array = np.array(list(audio_buffer))[-model_frame_size:]
            
            # Convert to tensor (Silero VAD expects float32)
            audio_tensor = torch.from_numpy(audio_array).float()
            
            # Add batch dimension if needed
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Get speech probability
            speech_prob = model(audio_tensor, sample_rate).item()
            
            # Hysteresis to avoid flickering
            if speech_prob >= 0.5:
                speech_frames += 1
                silence_frames = 0
                if speech_frames >= min_speech_frames and not speech_active:
                    speech_active = True
                    print(f"\r🎤 SPEECH DETECTED (confidence: {speech_prob:.3f})                          ")
            else:
                silence_frames += 1
                speech_frames = 0
                if silence_frames >= min_silence_frames and speech_active:
                    speech_active = False
                    print(f"\r🔇 SILENCE           (confidence: {speech_prob:.3f})                          ")
            
            # Show current status with probability
            status_icon = "🎤" if speech_active else "🔇"
            status_text = "SPEECH" if speech_active else "SILENCE"
            confidence_bar = "█" * int(speech_prob * 20) + "░" * (20 - int(speech_prob * 20))
            print(f'\r{status_icon} {status_text} [{confidence_bar}] {speech_prob:.3f}   ', end='')
    
    # Start microphone stream
    print("🎙️  Monitoring microphone...")
    print("📊 Speech detection is active")
    print("Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        with sd.InputStream(
            callback=callback,
            channels=1,  # Mono
            samplerate=sample_rate,
            blocksize=model_frame_size  # Match VAD frame size
        ):
            input()
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped.")
    
if __name__ == "__main__":
    realtime_vad_monitor()