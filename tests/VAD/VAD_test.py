# VAD_test.py
import argparse
import soundfile as sf
import torch
from pathlib import Path
from silero_vad import load_silero_vad, get_speech_timestamps


def process_audio(input_path, output_dir=None):
    """
    Process audio file with VAD and save speech segments
    
    Args:
        input_path: Path to audio file
        output_dir: Directory to save segments (default: current/segments)
    """
    # Convert to Path object
    input_path = Path(input_path)
    
    # Validate input file exists
    if not input_path.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")
    
    # Set default output directory to current/segments
    if output_dir is None:
        current = Path(__file__).resolve().parent
        output_dir = current / "segments"
    else:
        output_dir = Path(output_dir)
    
    # Load everything
    print(f"📁 Loading audio: {input_path}")
    audio, sr = sf.read(input_path)
    
    # Convert to mono if stereo
    wav = torch.from_numpy(audio if len(audio.shape) == 1 else audio.mean(axis=1)).float()
    
    # Load VAD model
    print("🤖 Loading VAD model...")
    model = load_silero_vad()
    
    # Get speech segments
    print("🎯 Detecting speech segments...")
    segments = get_speech_timestamps(wav, model, return_seconds=True)
    
    if not segments:
        print("⚠️ No speech detected in the audio file")
        return []
    
    # Create output folder
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each segment
    saved_files = []
    for i, seg in enumerate(segments, 1):
        start_sample = int(seg['start'] * sr)
        end_sample = int(seg['end'] * sr)
        
        output_file = output_dir / f"segment_{i:03d}.wav"
        sf.write(output_file, audio[start_sample:end_sample], sr)
        saved_files.append(output_file)
        
        print(f"  ✅ Segment {i:03d}: {seg['start']:.2f}s - {seg['end']:.2f}s ({seg['end'] - seg['start']:.2f}s)")
    
    print(f"\n✅ Saved {len(segments)} segments to '{output_dir}/'")
    return saved_files


def main():
    parser = argparse.ArgumentParser(
        description="Voice Activity Detection (VAD) - Extract speech segments from audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python VAD_test.py audio.opus
  python VAD_test.py audio.opus -o my_segments
  python VAD_test.py audio.opus --output my_segments
        """
    )
    
    parser.add_argument(
        "audio_path",
        help="Path to the audio file (supports: wav, mp3, opus, flac, etc.)"
    )
    
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory for segments (default: current/segments)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed output"
    )
    
    args = parser.parse_args()
    
    try:
        process_audio(args.audio_path, args.output)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())