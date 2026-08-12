"""Automatic Speech Recognition engine using Shenava-Koochik ONNX."""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import onnxruntime as ort

from ..config import config


class ASREngine:
    """Shenava-Koochik v1.0 ASR using ONNX Runtime."""

    def __init__(self):
        # Model paths
        self.model_path = config.get_path("model_paths.asr_onnx")
        self.tokens_path = config.get_path("model_paths.asr_tokens")
        self.mel_filters_path = config.get_path("model_paths.asr_mel_filters")

        # Load tokens
        with open(self.tokens_path, "r", encoding="utf-8") as f:
            token_data = json.load(f)

        if isinstance(token_data, dict):
            self.tokens = token_data["tokens"]
            self.blank_id = int(token_data.get("blank_id", 1024))
        else:
            self.tokens = token_data
            self.blank_id = 1024

        # Load mel filters
        with open(self.mel_filters_path, "r", encoding="utf-8") as f:
            mel_filter_data = json.load(f)

        self.mel_filters = np.asarray(mel_filter_data, dtype=np.float32)

        if self.mel_filters.shape != (80, 257):
            raise ValueError(
                f"Unexpected mel filter shape: {self.mel_filters.shape}; expected (80, 257)"
            )

        # ONNX session
        self.session = ort.InferenceSession(str(self.model_path))

        # Inspect model inputs/outputs
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if config.get("general.debug_asr", False):
            print("\nASR ONNX model:")
            for inp in inputs:
                print(f"  INPUT  {inp.name}: {inp.shape} {inp.type}")
            for out in outputs:
                print(f"  OUTPUT {out.name}: {out.shape} {out.type}")

        self.signal_input_name = "processed_signal"
        self.length_input_name = "processed_signal_length"

        input_names = {inp.name for inp in inputs}

        if self.signal_input_name not in input_names:
            raise RuntimeError(
                f"ONNX model does not contain '{self.signal_input_name}'. "
                f"Available inputs: {sorted(input_names)}"
            )

        if self.length_input_name not in input_names:
            raise RuntimeError(
                f"ONNX model does not contain '{self.length_input_name}'. "
                f"Available inputs: {sorted(input_names)}"
            )

        # Shenava / NeMo preprocessing configuration
        self.sample_rate = 16000
        self.n_fft = 512
        self.win_length = 400
        self.hop_length = 160
        self.n_mels = 80
        self.preemphasis = 0.97
        self.center_pad = 256
        self.fixed_frames = 2005
        self.log_guard = 5.960464477539063e-08
        self.output_stride = 8

        # Input dtype
        first_input = inputs[0]
        if "float16" in first_input.type:
            self.input_dtype = np.float16
        else:
            self.input_dtype = np.float32

        # Hann window
        self.window = np.hanning(self.win_length).astype(np.float32)

        # Transcript storage - timestamped directory like VAD
        base_dir = Path(config.get("asr.transcripts_dir", "assets/transcripts"))
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.transcripts_dir = base_dir / self.session_timestamp
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)

        self.all_transcripts = []
        self.transcript_summary = []

        if config.get("general.debug_asr", False):
            print(f"  ASR sample rate: {self.sample_rate}")
            print(f"  ASR blank ID: {self.blank_id}")
            print(f"  ASR input dtype: {self.input_dtype}")
            print(f"  Transcripts dir: {self.transcripts_dir}")
            print("  ASR preprocessing: Shenava/NeMo compatible")

    # =============================================================
    # AUDIO PREPROCESSING
    # =============================================================

    def _apply_preemphasis(self, audio: np.ndarray) -> np.ndarray:
        """Apply NeMo-compatible preemphasis."""
        if len(audio) <= 1:
            return audio

        output = np.empty_like(audio)
        output[0] = audio[0]
        output[1:] = audio[1:] - self.preemphasis * audio[:-1]
        return output

    def _compute_mel(self, audio: np.ndarray):
        """Compute mel spectrogram matching Shenava-Koochik preprocessor configuration."""
        audio = np.asarray(audio, dtype=np.float32)

        if audio.size == 0:
            raise ValueError("Cannot compute mel spectrogram from empty audio.")

        audio = self._apply_preemphasis(audio)

        if len(audio) > 1:
            audio = np.pad(audio, (self.center_pad, self.center_pad), mode="reflect")
        else:
            audio = np.pad(audio, (self.center_pad, self.center_pad), mode="constant")

        original_num_samples = len(audio) - 2 * self.center_pad
        actual_frames = max(
            1,
            min(
                self.fixed_frames,
                original_num_samples // self.hop_length + 1
            )
        )

        mel_spec = np.zeros((self.n_mels, actual_frames), dtype=np.float32)

        for frame_index in range(actual_frames):
            start = frame_index * self.hop_length
            frame = audio[start:start + self.win_length]

            if len(frame) < self.win_length:
                frame = np.pad(
                    frame,
                    (0, self.win_length - len(frame)),
                    mode="constant",
                    constant_values=0.0
                )

            frame = frame * self.window
            spectrum = np.fft.rfft(frame, n=self.n_fft)
            power = (np.abs(spectrum) ** 2).astype(np.float32)
            mel_spec[:, frame_index] = self.mel_filters @ power

        mel_spec = np.log(mel_spec + self.log_guard)
        return mel_spec, actual_frames

    # =============================================================
    # CTC DECODER
    # =============================================================

    def _decode_ctc(self, logits: np.ndarray) -> str:
        """Greedy CTC decoder."""
        token_ids = np.argmax(logits, axis=-1)
        decoded = []
        previous = self.blank_id

        for token_id in token_ids:
            token_id = int(token_id)

            if token_id == self.blank_id:
                previous = token_id
                continue

            if token_id == previous:
                continue

            if not (0 <= token_id < len(self.tokens)):
                previous = token_id
                continue

            token = self.tokens[token_id]

            if token.startswith("<") and token.endswith(">"):
                previous = token_id
                continue

            decoded.append(token)
            previous = token_id

        text = "".join(decoded)
        text = text.replace("▁", " ").strip()
        return text

    # =============================================================
    # TRANSCRIPTION
    # =============================================================

    def transcribe(self, audio_data: np.ndarray, segment_num: int) -> dict:
        """Transcribe one complete audio segment."""
        audio_array = np.asarray(audio_data, dtype=np.float32)

        if audio_array.size == 0:
            return {"text": "[EMPTY]"}

        if config.get("general.debug_asr", False):
            duration = len(audio_array) / self.sample_rate
            rms = float(np.sqrt(np.mean(audio_array ** 2)))
            print("\n")
            print("=" * 60)
            print(f"ASR segment {segment_num}")
            print("=" * 60)
            print(f"Audio samples : {len(audio_array)}")
            print(f"Audio duration: {duration:.3f}s")
            print(f"Audio min/max : {audio_array.min():.5f} / {audio_array.max():.5f}")
            print(f"Audio RMS     : {rms:.5f}")

        try:
            mel, actual_frames = self._compute_mel(audio_array)

            if config.get("general.debug_asr", False):
                print(f"Mel shape     : {mel.shape}")
                print(f"Actual frames : {actual_frames}")
                print(f"Mel min/max   : {mel.min():.5f} / {mel.max():.5f}")
                print(f"Mel mean/std  : {mel.mean():.5f} / {mel.std():.5f}")

            if mel.shape[1] > self.fixed_frames:
                mel = mel[:, :self.fixed_frames]
                actual_frames = self.fixed_frames
            elif mel.shape[1] < self.fixed_frames:
                mel = np.pad(
                    mel,
                    ((0, 0), (0, self.fixed_frames - mel.shape[1])),
                    mode="constant",
                    constant_values=0.0
                )

            input_tensor = mel[np.newaxis, :, :].astype(self.input_dtype)
            input_length = np.array([actual_frames], dtype=np.int64)

            if config.get("general.debug_asr", False):
                print(f"ONNX input    : {input_tensor.shape}")
                print(f"Input length  : {input_length.tolist()}")

            outputs = self.session.run(
                None,
                {
                    self.signal_input_name: input_tensor,
                    self.length_input_name: input_length
                }
            )

            logits = outputs[0][0]
            encoded_lengths = outputs[1]
            encoded_length = int(encoded_lengths[0])

            if config.get("general.debug_asr", False):
                print(f"Logits shape  : {logits.shape}")
                print(f"Encoded length: {encoded_length}")

            usable_steps = min(encoded_length, logits.shape[0])
            logits = logits[:usable_steps]

            token_ids = np.argmax(logits, axis=-1)

            if config.get("general.debug_asr", False):
                blank_percentage = float(np.mean(token_ids == self.blank_id))
                print(f"Usable steps  : {usable_steps}")
                print(f"Blank ratio   : {blank_percentage:.2%}")
                print(f"Token IDs     : {token_ids[:40].tolist()}")

            text = self._decode_ctc(logits)

            if not text:
                text = "[EMPTY]"

            if config.get("general.debug_asr", False):
                print(f"Transcript    : {text}")

        except Exception as e:
            text = f"[Error: {type(e).__name__}: {e}]"
            print(f"ASR ERROR: {text}")

        # Save segment transcript
        txt_filename = f"segment_{segment_num:04d}.txt"
        txt_path = self.transcripts_dir / txt_filename

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        # Store transcript
        self.all_transcripts.append(text)
        timestamp = datetime.now().strftime("%H:%M:%S")

        self.transcript_summary.append({
            "number": segment_num,
            "time": timestamp,
            "text": text
        })

        return {
            "text": text,
            "txt_path": txt_path
        }

    # =============================================================
    # SAVE TRANSCRIPTS
    # =============================================================

    def save_combined_files(self):
        """Save all transcripts to combined files."""
        if not self.all_transcripts:
            return

        # Full text
        combined_path = self.transcripts_dir / f"all_transcripts_{self.session_timestamp}.txt"

        with open(combined_path, "w", encoding="utf-8") as f:
            f.write("=" * 80)
            f.write("\n")
            f.write(f"Session: {self.session_timestamp}\n")
            f.write(f"Total segments: {len(self.all_transcripts)}\n")
            f.write("=" * 80)
            f.write("\n\n")

            for segment in self.transcript_summary:
                f.write(f"[{segment['time']}] Segment {segment['number']:04d}\n")
                f.write(f"Transcript: {segment['text']}\n")
                f.write("-" * 80)
                f.write("\n\n")

        # Plain transcript
        plain_path = self.transcripts_dir / f"all_transcripts_plain_{self.session_timestamp}.txt"

        with open(plain_path, "w", encoding="utf-8") as f:
            for i, text in enumerate(self.all_transcripts, 1):
                f.write(f"Segment {i:04d}: {text}\n\n")

        # JSON
        json_path = self.transcripts_dir / f"all_transcripts_{self.session_timestamp}.json"

        json_data = {
            "session_id": self.session_timestamp,
            "timestamp": datetime.now().isoformat(),
            "total_segments": len(self.all_transcripts),
            "segments": self.transcript_summary
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        print(f"\nTranscripts saved to: {self.transcripts_dir}")