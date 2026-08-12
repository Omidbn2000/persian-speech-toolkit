# src/myproject/tts/engine.py
import wave
from pathlib import Path
from io import BytesIO
from typing import Optional, Union
import numpy as np
import sounddevice as sd
from piper import PiperVoice

# Use relative import instead of absolute
from ..config import config  # .. goes up one level to myproject/


class TTSEngine:
    """Simple TTS engine using Piper for text-to-speech synthesis"""
    
    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        """
        Initialize TTS engine with a Piper voice model
        
        Args:
            model_path: Path to the .onnx model file.
                       If None, uses path from config.
        """
        # If no model_path provided, get it from config
        if model_path is None:
            model_path = config.get_path('model_paths.tts_onnx')
            if model_path is None:
                raise ValueError("No model path provided and no path found in config")
        
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at: {self.model_path}")
        
        try:
            self.voice = PiperVoice.load(self.model_path)
            self.sample_rate = self.voice.config.sample_rate
        except Exception as e:
            raise RuntimeError(f"Failed to load Piper model: {e}")
    
    def speak(self, text: str, save_path: Optional[Union[str, Path]] = None, 
              play: bool = True) -> Optional[Path]:
        """
        Convert text to speech, optionally save and/or play
        
        Args:
            text: Text to synthesize
            save_path: Path to save the audio file (directory or full path)
                      If directory, auto-generates filename
                      If None, doesn't save
            play: Whether to play the audio (default: True)
        
        Returns:
            Path to saved file if saved, None otherwise
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Synthesize to memory buffer
        buffer = BytesIO()
        try:
            with wave.open(buffer, 'wb') as wav_file:
                wav_file.setframerate(self.sample_rate)
                wav_file.setsampwidth(2)
                wav_file.setnchannels(1)
                self.voice.synthesize_wav(text, wav_file)
        except Exception as e:
            raise RuntimeError(f"Synthesis failed: {e}")
        
        # Play audio if requested
        if play:
            try:
                buffer.seek(0)
                audio_data = np.frombuffer(buffer.read(), dtype=np.int16)
                sd.play(audio_data, samplerate=self.sample_rate)
                sd.wait()
            except Exception as e:
                print(f"Warning: Audio playback failed: {e}")
        
        # Save audio if save_path is provided
        saved_path = None
        if save_path:
            save_path = Path(save_path)
            
            # If it's a directory, auto-generate filename
            if save_path.is_dir() or str(save_path).endswith('/') or str(save_path).endswith('\\'):
                save_path.mkdir(parents=True, exist_ok=True)
                # Use first few words for filename
                safe_name = "".join(c for c in text[:30] if c.isalnum() or c in (' ', '-', '_')).strip()
                if not safe_name:
                    safe_name = "tts_output"
                final_path = save_path / f"{safe_name}.wav"
            else:
                # It's a full file path
                final_path = save_path
                final_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write buffer to file
            try:
                buffer.seek(0)
                with wave.open(str(final_path), 'wb') as wav_file:
                    wav_file.setframerate(self.sample_rate)
                    wav_file.setsampwidth(2)
                    wav_file.setnchannels(1)
                    wav_file.writeframes(buffer.read())
                
                saved_path = final_path
                print(f"File saved to: {final_path}")
            except Exception as e:
                print(f"Warning: Failed to save audio: {e}")
        
        return saved_path
    
    def speak_multiple(self, texts: list, save_dir: Optional[Union[str, Path]] = None,
                       play: bool = False) -> list:
        """Synthesize multiple texts"""
        saved_files = []
        for i, text in enumerate(texts):
            if save_dir:
                save_path = Path(save_dir) / f"tts_output_{i:03d}.wav"
            else:
                save_path = None
            
            file_path = self.speak(text, save_path, play)
            if file_path:
                saved_files.append(file_path)
        
        return saved_files