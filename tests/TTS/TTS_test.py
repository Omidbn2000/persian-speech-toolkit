# tts_engine.py
import wave
from pathlib import Path
from io import BytesIO
import numpy as np
import sounddevice as sd
from piper import PiperVoice


class TTSEngine:
    """Simple TTS engine using Piper for text-to-speech synthesis"""
    
    def __init__(self, model_path):
        """
        Initialize TTS engine with a Piper voice model
        
        Args:
            model_path: Path to the .onnx model file
        """
        self.model_path = Path(model_path)
        self.voice = PiperVoice.load(self.model_path)
        self.sample_rate = self.voice.config.sample_rate
    
    def speak(self, text, save_path=None, play=True):
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
        # Synthesize to memory buffer
        buffer = BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setframerate(self.sample_rate)
            wav_file.setsampwidth(2)
            wav_file.setnchannels(1)
            self.voice.synthesize_wav(text, wav_file)
        
        # Play audio if requested
        if play:
            buffer.seek(0)
            audio_data = np.frombuffer(buffer.read(), dtype=np.int16)
            sd.play(audio_data, samplerate=self.sample_rate)
            sd.wait()
        
        # Save audio if save_path is provided
        saved_path = None
        if save_path:
            save_path = Path(save_path)
            
            # If it's a directory, auto-generate filename
            if save_path.is_dir():
                save_path.mkdir(parents=True, exist_ok=True)
                final_path = save_path / "tts_output.wav"
            else:
                # It's a full file path
                final_path = save_path
                final_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write buffer to file
            buffer.seek(0)
            with wave.open(str(final_path), 'wb') as wav_file:
                wav_file.setframerate(self.sample_rate)
                wav_file.setsampwidth(2)
                wav_file.setnchannels(1)
                wav_file.writeframes(buffer.read())
            
            saved_path = final_path
            print ("file saved in "+str(save_path))
        return saved_path


# ============================================================
# MAIN - Usage Examples
# ============================================================

if __name__ == "__main__":
    # Get current directory and project root
    current = Path(__file__).resolve().parent
    PROJECT_ROOT = current.parent.parent.parent
    
    # Initialize TTS engine
    tts = TTSEngine(
        PROJECT_ROOT / "models/Mana-Persian-Piper/fa_IR-mana-medium.onnx"
    )
    
    # Example 1: Play and save to current/outputs/
    tts.speak(
        text="سلام! این یک تست است.",
        save_path=current / "outputs/a.wav"
    )
    
    # Example 2: Save only (no playback)
    tts.speak(
        text="این فایل فقط ذخیره می‌شود.",
        save_path=current / "outputs/b.wav",
        play=False
    )
    
    # Example 3: Play only (no save)
    tts.speak(
        text="این فقط پخش می‌شود.",
        save_path=None
    )