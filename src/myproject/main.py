"""Main entry point - Automatic Speech Processing with VAD."""

import sys
import time
import threading
import queue
from collections import deque

import numpy as np
import sounddevice as sd
import keyboard

from silero_vad import load_silero_vad

from .tts import TTSEngine
from .vad import VADEngine
from .asr import ASREngine
from .config import config


# ================================================================
# LOADING
# ================================================================

def show_loading():
    """Load all speech-processing modules."""

    print("\n" + "=" * 60)
    print("  Initializing Speech Processing System")
    print("=" * 60)

    # ------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------

    print(
        "  Loading TTS Engine...",
        end=" ",
        flush=True
    )

    tts_engine = TTSEngine()

    print("✓")

    # ------------------------------------------------------------
    # VAD
    # ------------------------------------------------------------

    print(
        "  Loading VAD Engine...",
        end=" ",
        flush=True
    )

    vad_model = load_silero_vad()

    vad_engine = VADEngine(
        vad_model
    )

    print("✓")

    # ------------------------------------------------------------
    # ASR
    # ------------------------------------------------------------

    print(
        "  Loading ASR Engine...",
        end=" ",
        flush=True
    )

    asr_engine = ASREngine()

    print("✓")

    print("=" * 60)
    print("  All modules loaded successfully!")
    print("=" * 60)

    return (
        tts_engine,
        vad_engine,
        asr_engine
    )


# ================================================================
# HELP
# ================================================================

def show_help():
    """Display help menu."""

    print()
    print("=" * 60)
    print("  HELP - Available Commands")
    print("=" * 60)

    print(
        "  [S]      Toggle automatic speech recognition"
    )

    print(
        "  [Space]  Pause / Resume"
    )

    print(
        "  [Q]      Exit"
    )

    print("=" * 60)

    print()
    print(
        "  VAD automatically detects speech."
    )

    print(
        "  Just speak normally when S is ON."
    )

    print()


# ================================================================
# MAIN LOOP
# ================================================================

def run_main_loop(
    tts_engine,
    vad_engine,
    asr_engine
):
    """
    Main real-time speech loop.

    VAD controls segmentation automatically.

    Important architecture:

        Audio callback
            ↓
        VAD
            ↓
        speech buffer
            ↓
        completed segment queue
            ↓
        ASR worker
            ↓
        transcript

    ASR inference NEVER runs inside the audio callback.
    """

    # ------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------

    sample_rate = config.get(
        "vad.sample_rate",
        16000
    )

    block_size = config.get(
        "vad.frame_size",
        512
    )

    # ------------------------------------------------------------
    # VAD segmentation settings
    # ------------------------------------------------------------

    # Keep audio BEFORE VAD detects speech.
    #
    # 0.30 sec × 16000 = 4800 samples
    #
    pre_roll_seconds = 0.30

    pre_roll_samples = int(
        pre_roll_seconds
        * sample_rate
    )

    # Require a little continuous speech before
    # declaring that a real speech segment started.
    #
    # 2 × 32ms = ~64ms
    speech_start_blocks = 2

    # Do not immediately end speech when VAD says silence.
    #
    # 12 × 32ms = ~384ms
    #
    # This is the "hangover".
    silence_end_blocks = 12

    # Ignore extremely short segments.
    minimum_segment_seconds = 0.50

    minimum_segment_samples = int(
        minimum_segment_seconds
        * sample_rate
    )

    # ------------------------------------------------------------
    # State
    # ------------------------------------------------------------

    enabled = True

    paused = False

    is_speaking = False

    speech_candidate_blocks = 0

    silence_blocks = 0

    segment_count = 0

    # ------------------------------------------------------------
    # Buffers
    # ------------------------------------------------------------

    # Rolling audio before speech starts.
    pre_roll = deque(
        maxlen=pre_roll_samples
    )

    # Current speech segment.
    speech_buffer = []

    buffer_lock = threading.Lock()

    # ------------------------------------------------------------
    # Completed ASR segments
    #
    # The audio callback puts completed audio here.
    #
    # The main thread/worker performs ASR.
    # ------------------------------------------------------------

    completed_segments = queue.Queue()

    # ------------------------------------------------------------
    # ASR worker state
    # ------------------------------------------------------------

    worker_running = True

    # ------------------------------------------------------------
    # ASR worker
    # ------------------------------------------------------------

    def asr_worker():
        """
        Process completed speech segments.

        This runs outside the sounddevice callback.
        """

        while worker_running:

            try:

                item = completed_segments.get(
                    timeout=0.1
                )

            except queue.Empty:
                continue

            if item is None:
                break

            segment_number, audio = item

            duration = (
                len(audio)
                / sample_rate
            )

            print(
                "\n\n>>> "
                f"Transcribing segment "
                f"{segment_number} "
                f"({duration:.2f}s)..."
            )

            try:

                result = (
                    asr_engine.transcribe(
                        audio,
                        segment_number
                    )
                )

                print()
                print("=" * 60)
                print(
                    f"  RESULT "
                    f"[Segment {segment_number}]"
                )
                print("=" * 60)

                print(
                    f"  {result['text']}"
                )

                print("=" * 60)

            except Exception as e:

                print(
                    "\n>>> ASR worker error:"
                )

                print(
                    f"    {type(e).__name__}: {e}"
                )

            finally:

                completed_segments.task_done()

    # ------------------------------------------------------------
    # Start ASR worker
    # ------------------------------------------------------------

    worker_thread = threading.Thread(
        target=asr_worker,
        daemon=True
    )

    worker_thread.start()

    # ============================================================
    # AUDIO CALLBACK
    # ============================================================

    def audio_callback(
        indata,
        frames,
        time_info,
        status
    ):
        nonlocal \
            is_speaking, \
            speech_candidate_blocks, \
            silence_blocks, \
            speech_buffer, \
            segment_count

        # --------------------------------------------------------
        # Audio status
        # --------------------------------------------------------

        if status:

            status_text = str(status)

            if (
                "overflow"
                not in status_text.lower()
            ):

                print(
                    f"\nAudio status: "
                    f"{status_text}"
                )

        # --------------------------------------------------------
        # Pause
        # --------------------------------------------------------

        if paused:
            return

        # --------------------------------------------------------
        # Audio
        # --------------------------------------------------------

        audio = (
            indata[:, 0]
            .astype(np.float32)
            .copy()
        )

        # --------------------------------------------------------
        # Amplitude
        # --------------------------------------------------------

        rms = float(
            np.sqrt(
                np.mean(
                    audio ** 2
                )
            )
        )

        level = min(
            int(rms * 2000),
            30
        )

        amp_bar = (
            "█" * level
            + "░" * (30 - level)
        )

        # --------------------------------------------------------
        # Maintain pre-roll
        #
        # ALWAYS maintain this.
        #
        # This is what protects the beginning of speech.
        # --------------------------------------------------------

        pre_roll.extend(
            audio.tolist()
        )

        # --------------------------------------------------------
        # VAD
        # --------------------------------------------------------

        try:

            vad_result = (
                vad_engine.process(
                    indata
                )
            )

            is_speech = bool(
                vad_result["is_speech"]
            )

            probability = float(
                vad_result.get(
                    "probability",
                    0.0
                )
            )

        except Exception as e:

            print(
                f"\nVAD error: {e}"
            )

            return

        # --------------------------------------------------------
        # VAD display
        # --------------------------------------------------------

        vad_level = min(
            int(probability * 30),
            30
        )

        vad_bar = (
            "█" * vad_level
            + "░" * (30 - vad_level)
        )

        if is_speech:

            vad_label = "SPEECH"

        else:

            vad_label = "SILENCE"

        # ========================================================
        # SPEECH DETECTION
        # ========================================================

        if is_speech:

            # ----------------------------------------------------
            # Speech detected
            # ----------------------------------------------------

            speech_candidate_blocks += 1

            # Speech is back, so silence counter resets.
            silence_blocks = 0

            # ----------------------------------------------------
            # Start speech after confirmation
            # ----------------------------------------------------

            if not is_speaking:

                if (
                    speech_candidate_blocks
                    >= speech_start_blocks
                ):

                    is_speaking = True

                    segment_count += 1

                    # ------------------------------------------------
                    # IMPORTANT:
                    #
                    # Start with pre-roll.
                    #
                    # This means audio immediately BEFORE
                    # VAD detected speech is included.
                    # ------------------------------------------------

                    speech_buffer = list(
                        pre_roll
                    )

                    # Add current block.
                    speech_buffer.extend(
                        audio.tolist()
                    )

            else:

                # ------------------------------------------------
                # Already recording speech.
                # ------------------------------------------------

                speech_buffer.extend(
                    audio.tolist()
                )

        # ========================================================
        # SILENCE
        # ========================================================

        else:

            # ----------------------------------------------------
            # No speech
            # ----------------------------------------------------

            speech_candidate_blocks = 0

            # ----------------------------------------------------
            # Currently inside speech segment?
            # ----------------------------------------------------

            if is_speaking:

                # ------------------------------------------------
                # KEEP silence temporarily.
                #
                # This is the hangover buffer.
                # ------------------------------------------------

                speech_buffer.extend(
                    audio.tolist()
                )

                silence_blocks += 1

                # ------------------------------------------------
                # Only finish after enough silence.
                # ------------------------------------------------

                if (
                    silence_blocks
                    >= silence_end_blocks
                ):

                    # --------------------------------------------
                    # End speech segment
                    # --------------------------------------------

                    is_speaking = False

                    silence_blocks = 0

                    # --------------------------------------------
                    # Copy complete segment
                    # --------------------------------------------

                    segment_audio = np.asarray(
                        speech_buffer,
                        dtype=np.float32
                    )

                    speech_buffer = []

                    # --------------------------------------------
                    # Check minimum duration
                    # --------------------------------------------

                    if (
                        len(segment_audio)
                        >= minimum_segment_samples
                    ):

                        # ----------------------------------------
                        # Send to ASR worker.
                        #
                        # NEVER run ASR here.
                        # ----------------------------------------

                        completed_segments.put(
                            (
                                segment_count,
                                segment_audio
                            )
                        )

                    else:

                        print(
                            "\n>>> Ignored very short "
                            f"segment "
                            f"({len(segment_audio) / sample_rate:.2f}s)"
                        )

        # ========================================================
        # STATUS
        # ========================================================

        if not enabled:

            system_status = "OFF"

        elif paused:

            system_status = "PAUSED"

        else:

            system_status = "ON"

        if is_speaking:

            speech_status = "RECORDING"

        elif speech_candidate_blocks > 0:

            speech_status = "STARTING"

        else:

            speech_status = "WAITING"

        # Current segment duration
        with buffer_lock:

            current_duration = (
                len(speech_buffer)
                / sample_rate
            )

        print(
            f"\r  Mic [{amp_bar}] "
            f"| VAD [{vad_bar}] "
            f"{vad_label:<7} "
            f"| {speech_status:<9} "
            f"| {system_status:<6} "
            f"| {current_duration:5.1f}s",
            end="",
            flush=True
        )

    # ============================================================
    # MICROPHONE
    # ============================================================

    print(
        "\nOpening microphone..."
    )

    with sd.InputStream(
        callback=audio_callback,
        channels=1,
        samplerate=sample_rate,
        blocksize=block_size,
        dtype="float32"
    ):

        print(
            "Microphone ready."
        )

        print()
        print(
            "Automatic VAD is ACTIVE."
        )

        print(
            "Speak normally."
        )

        print(
            "Press [S] to disable/enable recognition."
        )

        print()

        # ========================================================
        # CONTROL LOOP
        # ========================================================

        try:

            while True:

                # ------------------------------------------------
                # Toggle automatic recognition
                # ------------------------------------------------

                if keyboard.is_pressed("s"):

                    enabled = not enabled

                    if not enabled:

                        # ----------------------------------------
                        # Reset segmentation state.
                        # ----------------------------------------

                        is_speaking = False

                        speech_candidate_blocks = 0

                        silence_blocks = 0

                        speech_buffer = []

                        print(
                            "\n\n>>> "
                            "Automatic speech recognition: OFF"
                        )

                    else:

                        print(
                            "\n\n>>> "
                            "Automatic speech recognition: ON"
                        )

                    time.sleep(0.35)

                # ------------------------------------------------
                # Pause
                # ------------------------------------------------

                elif keyboard.is_pressed(
                    "space"
                ):

                    paused = not paused

                    if paused:

                        print(
                            "\n\n>>> "
                            "Microphone PAUSED"
                        )

                    else:

                        print(
                            "\n\n>>> "
                            "Microphone RESUMED"
                        )

                    time.sleep(0.35)

                # ------------------------------------------------
                # Quit
                # ------------------------------------------------

                elif keyboard.is_pressed("q"):

                    print(
                        "\n\n>>> Exiting..."
                    )

                    break

                time.sleep(0.05)

        except KeyboardInterrupt:

            print(
                "\n\n>>> Interrupted."
            )

    # ============================================================
    # SHUTDOWN
    # ============================================================

    worker_running = False

    # Wake worker
    completed_segments.put(None)

    worker_thread.join(
        timeout=2.0
    )

    # Wait for queued ASR jobs
    #
    # Usually the queue will already be empty.
    # ============================================================

    try:

        completed_segments.join()

    except Exception:

        pass

    # ------------------------------------------------------------
    # Save transcripts
    # ------------------------------------------------------------

    asr_engine.save_combined_files()

    print(
        "\n>>> Shutdown complete."
    )


# ================================================================
# MAIN
# ================================================================

def main():
    """Application entry point."""

    try:

        (
            tts_engine,
            vad_engine,
            asr_engine
        ) = show_loading()

        show_help()

        run_main_loop(
            tts_engine,
            vad_engine,
            asr_engine
        )

    except Exception as e:

        print(
            "\nError:"
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        sys.exit(1)


if __name__ == "__main__":
    main()