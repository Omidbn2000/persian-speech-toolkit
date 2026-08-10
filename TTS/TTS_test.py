import wave
from piper import PiperVoice

# Load the voice model
# Make sure the path to the .onnx file is correct
voice = PiperVoice.load("./Mana-Persian-Piper/fa_IR-mana-medium.onnx")

# Synthesize speech and save it to a WAV file
with wave.open("persian_output.wav", "wb") as wav_file:
    voice.synthesize_wav("من از اون آسمون آبی میخوام... من از اون، شبهای مهتابی میخوام!", wav_file)