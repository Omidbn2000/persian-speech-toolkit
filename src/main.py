"""Main entry point - Automatic Speech Processing with VAD."""

import sys
import time
import threading
import queue
from collections import deque

import numpy as np
import sounddevice as sd

from .tts import TTSEngine
from .vad import VADEngine
from .asr import ASREngine
from .config import config


def show_loading():
    """Load all speech-processing modules."""
    print("\n" + "=" * 60)
    print("  Initializing Speech Processing System")
    print("=" * 60)

    print("  Loading TTS Engine...", end=" ", flush=True)
    tts_engine = TTSEngine()
    print("✓")

    print("  Loading VAD Engine...", end=" ", flush=True)
    vad_engine = VADEngine()
    print("✓")

    print("  Loading ASR Engine...", end=" ", flush=True)
    asr_engine = ASREngine()
    print("✓")

    print("=" * 60)
    print("  All modules loaded successfully!")
    print("=" * 60)

    return tts_engine, vad_engine, asr_engine


def run_main_loop(tts_engine, vad_engine, asr_engine):
    """Main real-time speech loop."""
    sample_rate = config.get("vad.sample_rate", 16000)
    block_size = config.get("vad.frame_size", 512)

    pre_roll_seconds = config.get("segmentation.pre_roll_seconds", 0.30)
    pre_roll_samples = int(pre_roll_seconds * sample_rate)
    speech_start_blocks = config.get("segmentation.speech_start_blocks", 2)
    silence_end_blocks = config.get("segmentation.silence_end_blocks", 12)
    minimum_segment_seconds = config.get("segmentation.minimum_segment_seconds", 0.50)
    minimum_segment_samples = int(minimum_segment_seconds * sample_rate)

    tts_enabled = config.get("general.tts_enabled", False)

    is_speaking = False
    speech_candidate_blocks = 0
    silence_blocks = 0
    segment_count = 0
    last_transcript = ""
    transcribing = False

    pre_roll = deque(maxlen=pre_roll_samples)
    speech_buffer = []
    buffer_lock = threading.Lock()
    completed_segments = queue.Queue()

    worker_running = True

    def asr_worker():
        nonlocal last_transcript, transcribing

        while worker_running:
            try:
                item = completed_segments.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                break

            segment_number, audio = item
            duration = len(audio) / sample_rate

            transcribing = True
            # Clear the entire status line fully (150 chars to be safe)
            print(f"\r{' ' * 150}", end="", flush=True)
            print(f"\r>>> Transcribing segment {segment_number} ({duration:.2f}s)...", flush=True)

            try:
                result = asr_engine.transcribe(audio, segment_number)
                text = result['text']
                last_transcript = text

                print(f"============================================================")
                print(f"  [Segment {segment_number}]:")
                print(f"  {text}")
                print(f"============================================================")

                if tts_enabled and text and text != "[EMPTY]" and not text.startswith("[Error"):
                    print(f"  Speaking...")
                    try:
                        tts_engine.speak(text, play=True)
                    except Exception as e:
                        print(f"  TTS error: {e}")

            except Exception as e:
                last_transcript = f"[Error]"
                print(f">>> ASR error: {type(e).__name__}: {e}")
            finally:
                completed_segments.task_done()
                transcribing = False
                print()  # blank line before status bar resumes

    worker_thread = threading.Thread(target=asr_worker, daemon=True)
    worker_thread.start()

    def audio_callback(indata, frames, time_info, status):
        nonlocal is_speaking, speech_candidate_blocks, silence_blocks, speech_buffer, segment_count

        if transcribing:
            return

        if status:
            status_text = str(status)
            if "overflow" not in status_text.lower():
                print(f"\nAudio status: {status_text}")

        audio = indata[:, 0].astype(np.float32).copy()

        rms = float(np.sqrt(np.mean(audio ** 2)))
        level = min(int(rms * 2000), 30)
        amp_bar = "█" * level + "░" * (30 - level)

        pre_roll.extend(audio.tolist())

        try:
            vad_result = vad_engine.process(indata)
            is_speech = bool(vad_result["is_speech"])
            probability = float(vad_result.get("probability", 0.0))
        except Exception:
            return

        vad_level = min(int(probability * 30), 30)
        vad_bar = "█" * vad_level + "░" * (30 - vad_level)
        vad_label = "SPEECH" if is_speech else "SILENCE"

        if is_speech:
            speech_candidate_blocks += 1
            silence_blocks = 0

            if not is_speaking:
                if speech_candidate_blocks >= speech_start_blocks:
                    is_speaking = True
                    segment_count += 1
                    speech_buffer = list(pre_roll)
                    speech_buffer.extend(audio.tolist())
            else:
                speech_buffer.extend(audio.tolist())
        else:
            speech_candidate_blocks = 0

            if is_speaking:
                speech_buffer.extend(audio.tolist())
                silence_blocks += 1

                if silence_blocks >= silence_end_blocks:
                    is_speaking = False
                    silence_blocks = 0

                    segment_audio = np.asarray(speech_buffer, dtype=np.float32)
                    speech_buffer = []

                    if len(segment_audio) >= minimum_segment_samples:
                        completed_segments.put((segment_count, segment_audio))

        if is_speaking:
            speech_status = "REC"
        elif speech_candidate_blocks > 0:
            speech_status = "START"
        else:
            speech_status = "WAIT"

        tts_status = "TTS:ON" if tts_enabled else "TTS:OFF"

        with buffer_lock:
            current_duration = len(speech_buffer) / sample_rate

        status_line = (
            f"\r  Mic[{amp_bar}] VAD[{vad_bar}] {vad_label} {speech_status} {tts_status} "
            f"{current_duration:.1f}s Seg:{segment_count} "
        )
        # Pad generously to wipe any previous longer line
        status_line = status_line.ljust(150)

        print(status_line, end="", flush=True)

    print("\nOpening microphone...")

    with sd.InputStream(
        callback=audio_callback,
        channels=1,
        samplerate=sample_rate,
        blocksize=block_size,
        dtype="float32"
    ):
        print("Microphone ready. Speak normally. Press Ctrl+C to exit.")
        print()

        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\n>>> Exiting...")

    worker_running = False
    completed_segments.put(None)
    worker_thread.join(timeout=1.0)
    asr_engine.save_combined_files()


def main():
    """Application entry point."""
    try:
        tts_engine, vad_engine, asr_engine = show_loading()
        run_main_loop(tts_engine, vad_engine, asr_engine)
    except Exception as e:
        print("\nError:")
        print(f"{type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()