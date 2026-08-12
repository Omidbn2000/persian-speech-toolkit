"""Voice Activity Detection engine using Silero VAD"""
import wave
import time
import numpy as np
import torch
from pathlib import Path
from collections import deque
from datetime import datetime
from ..config import config
from silero_vad import load_silero_vad

class VADEngine:
    """Real-time Voice Activity Detection"""
    
    def __init__(self):
        self.model = load_silero_vad()
        self.sample_rate = config.get('vad.sample_rate', 16000)
        self.frame_size = config.get('vad.frame_size', 512)
        self.threshold = config.get('vad.threshold', 0.5)
        self.min_speech_frames = config.get('vad.min_speech_frames', 3)
        self.min_silence_frames = config.get('vad.min_silence_frames', 5)
        self.buffer_duration = config.get('vad.buffer_duration', 1.0)
        
        # Saving config
        self.save_audio = config.get('vad.save_audio', True)
        self.min_speech_duration = config.get('vad.min_speech_duration', 0.5)
        self.max_silence_duration = config.get('vad.max_silence_duration', 0.5)
        self.audio_format = config.get('vad.audio_format', 'wav')
        
        buffer_samples = int(self.sample_rate * self.buffer_duration)
        self.audio_buffer = deque(maxlen=buffer_samples)
        self.speech_buffer = []
        
        self.speech_active = False
        self.speech_frames = 0
        self.silence_frames = 0
        self.current_prob = 0.0
        
        # Timing
        self.speech_start_time = None
        self.silence_start_time = None
        
        # Saving
        self.save_dir = None
        self.segment_count = 0
        self._init_save_dir()
    
    def _init_save_dir(self):
        """Create unique directory for this session"""
        if self.save_audio:
            base_dir = Path(config.get('vad.save_dir', 'assets/audios/vad_segments'))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_dir = base_dir / timestamp
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def process(self, audio_chunk: np.ndarray) -> dict:
        """Process audio chunk and return VAD state"""
        chunk_flat = audio_chunk.flatten()
        self.audio_buffer.extend(chunk_flat)
        
        if len(self.audio_buffer) >= self.frame_size:
            audio_array = np.array(list(self.audio_buffer))[-self.frame_size:]
            audio_tensor = torch.from_numpy(audio_array).float()
            
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            speech_prob = self.model(audio_tensor, self.sample_rate).item()
            self.current_prob = speech_prob
            
            if speech_prob >= self.threshold:
                self.speech_frames += 1
                self.silence_frames = 0
                self.silence_start_time = None
                
                if not self.speech_active:
                    self.speech_start_time = time.time()
                
                if self.speech_frames >= self.min_speech_frames:
                    self.speech_active = True
            else:
                self.silence_frames += 1
                self.speech_frames = 0
                
                if self.silence_start_time is None:
                    self.silence_start_time = time.time()
                
                silence_duration = time.time() - self.silence_start_time if self.silence_start_time else 0
                
                if self.silence_frames >= self.min_silence_frames and silence_duration >= self.max_silence_duration:
                    if self.speech_active:
                        self._save_segment()
                    self.speech_active = False
                    self.speech_start_time = None
            
            if self.speech_active:
                self.speech_buffer.extend(chunk_flat)
        
        return {
            'is_speech': self.speech_active,
            'probability': self.current_prob
        }
    
    def _save_segment(self):
        """Save speech segment to file if conditions met"""
        if not self.save_audio or not self.speech_buffer:
            self.speech_buffer = []
            return
        
        duration = len(self.speech_buffer) / self.sample_rate
        if duration < self.min_speech_duration:
            self.speech_buffer = []
            return
        
        filename = f"segment_{self.segment_count:04d}.{self.audio_format}"
        filepath = self.save_dir / filename
        
        audio_data = np.array(self.speech_buffer, dtype=np.float32)
        audio_data = (audio_data * 32767).astype(np.int16)
        
        with wave.open(str(filepath), 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        self.segment_count += 1
        self.speech_buffer = []
        if config.get("general.debug_vad", False):
            print(f"\n[VAD] Saved segment {self.segment_count}: {filepath} ({duration:.2f}s)")