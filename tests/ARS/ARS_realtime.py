import sounddevice as sd
import numpy as np
import torch
from silero_vad import load_silero_vad
from collections import deque
import nemo.collections.asr as nemo_asr
import os
import time
from datetime import datetime

def realtime_vad_asr():
    """Real-time Voice Activity Detection with ASR transcription and file saving"""
    
    # Load VAD model
    print("Loading Silero VAD model...")
    vad_model = load_silero_vad()
    print("✓ VAD model loaded!")
    
    # Load ASR model
    print("Loading Shenava Koochik ASR model...")
    MODEL_PATH = r"C:\Users\sejron\Desktop\Embedded_AI\Phase2\Shenava-Koochik-v1.0\shenava-koochik-1.0.nemo"
    asr_model = nemo_asr.models.ASRModel.restore_from(MODEL_PATH)
    asr_model.eval()
    
    # Configure ASR
    print("Configuring ASR decoder...")
    asr_model.change_decoding_strategy(decoder_type="ctc")
    asr_model.encoder.set_default_att_context_size([70, 13])
    print("✓ ASR model loaded and configured!")
    print("-" * 70)
    
    # Settings
    sample_rate = 16000
    model_frame_size = 512
    buffer_duration = 2.0
    buffer_samples = int(sample_rate * buffer_duration)
    min_speech_duration = 0.5
    max_silence_duration = 0.5
    min_speech_samples = int(sample_rate * min_speech_duration)
    max_silence_samples = int(sample_rate * max_silence_duration)
    
    # Audio buffers
    audio_buffer = deque(maxlen=buffer_samples)
    speech_buffer = []
    silence_counter = 0
    speech_counter = 0
    
    # State tracking
    is_speaking = False
    segment_count = 0
    
    # Create output folders
    OUTPUT_FOLDER = "transcripts"
    SEGMENTS_FOLDER = "recorded_segments"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(SEGMENTS_FOLDER, exist_ok=True)
    
    # Store all transcripts for combined file
    all_transcripts = []
    transcript_summary = []
    
    # Session timestamp for unique filenames
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def transcribe_and_save(audio_data, seg_num):
        """Transcribe audio data and save to files"""
        try:
            # Save audio segment
            audio_filename = f"segment_{seg_num:03d}.wav"
            audio_path = os.path.join(SEGMENTS_FOLDER, audio_filename)
            
            # Convert to proper format and save
            audio_array = np.array(audio_data, dtype=np.float32)
            
            # Normalize
            if np.max(np.abs(audio_array)) > 0:
                audio_array = audio_array / np.max(np.abs(audio_array))
            
            import soundfile as sf
            sf.write(audio_path, audio_array, sample_rate)
            
            # Transcribe
            results = asr_model.transcribe([audio_path], batch_size=1)
            text = results[0].text if hasattr(results[0], "text") else str(results[0])
            text = text.strip()
            
            if text == "":
                text = "[EMPTY TRANSCRIPT]"
            
            # Save individual transcript
            txt_filename = f"segment_{seg_num:03d}.txt"
            txt_path = os.path.join(OUTPUT_FOLDER, txt_filename)
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # Store for combined file
            all_transcripts.append(text)
            timestamp = datetime.now().strftime("%H:%M:%S")
            transcript_summary.append({
                'number': seg_num,
                'time': timestamp,
                'text': text,
                'audio_file': audio_filename
            })
            
            return text, audio_path, txt_path
            
        except Exception as e:
            error_text = f"[Error: {str(e)}]"
            all_transcripts.append(error_text)
            
            # Save error to file
            txt_filename = f"segment_{seg_num:03d}_error.txt"
            txt_path = os.path.join(OUTPUT_FOLDER, txt_filename)
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(error_text)
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            transcript_summary.append({
                'number': seg_num,
                'time': timestamp,
                'text': error_text,
                'audio_file': f"segment_{seg_num:03d}.wav"
            })
            
            return error_text, None, txt_path
    
    def save_combined_files():
        """Save all transcripts to combined files"""
        if not all_transcripts:
            return
        
        print("\n" + "=" * 70)
        print("💾 Saving combined transcript files...")
        print("=" * 70)
        
        # Combined file with metadata
        combined_filename = f"all_transcripts_{session_timestamp}.txt"
        combined_path = os.path.join(OUTPUT_FOLDER, combined_filename)
        
        with open(combined_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("REAL-TIME SPEECH RECOGNITION TRANSCRIPTS\n")
            f.write(f"Session: {session_timestamp}\n")
            f.write(f"Total segments: {len(all_transcripts)}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            for summary in transcript_summary:
                f.write(f"[{summary['time']}] Segment {summary['number']:03d}\n")
                f.write(f"Audio: {summary['audio_file']}\n")
                f.write(f"Transcript: {summary['text']}\n")
                f.write("-" * 80 + "\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF TRANSCRIPTS\n")
            f.write("=" * 80 + "\n")
        
        print(f"✓ Combined transcript with metadata: {combined_filename}")
        
        # Plain combined file (just text)
        plain_filename = f"all_transcripts_plain_{session_timestamp}.txt"
        plain_path = os.path.join(OUTPUT_FOLDER, plain_filename)
        
        with open(plain_path, 'w', encoding='utf-8') as f:
            for i, text in enumerate(all_transcripts, 1):
                f.write(f"Segment {i:03d}: {text}\n\n")
        
        print(f"✓ Plain combined transcript: {plain_filename}")
        
        # Continuous text file (all concatenated)
        continuous_filename = f"all_transcripts_continuous_{session_timestamp}.txt"
        continuous_path = os.path.join(OUTPUT_FOLDER, continuous_filename)
        
        with open(continuous_path, 'w', encoding='utf-8') as f:
            for text in all_transcripts:
                if not text.startswith("[Error") and text != "[EMPTY TRANSCRIPT]":
                    f.write(text + " ")
        
        print(f"✓ Continuous transcript: {continuous_filename}")
        
        # JSON format for programmatic use
        import json
        json_filename = f"all_transcripts_{session_timestamp}.json"
        json_path = os.path.join(OUTPUT_FOLDER, json_filename)
        
        json_data = {
            'session_id': session_timestamp,
            'timestamp': datetime.now().isoformat(),
            'total_segments': len(all_transcripts),
            'segments': transcript_summary
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ JSON transcript: {json_filename}")
        print(f"\n📁 All files saved to: {os.path.abspath(OUTPUT_FOLDER)}")
    
    def callback(indata, frames, time_info, status):
        nonlocal is_speaking, speech_buffer, silence_counter, speech_counter, segment_count
        
        if status:
            print(f"\nStatus: {status}")
        
        # Get audio chunk
        audio_chunk = indata.flatten()
        audio_buffer.extend(audio_chunk)
        
        # Process VAD when we have enough data
        if len(audio_buffer) >= model_frame_size:
            # Get exactly 512 samples for VAD
            audio_array = np.array(list(audio_buffer))[-model_frame_size:]
            
            # Convert to tensor for VAD
            audio_tensor = torch.from_numpy(audio_array).float()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Get speech probability
            speech_prob = vad_model(audio_tensor, sample_rate).item()
            
            # State machine for speech detection
            if speech_prob >= 0.5:
                speech_counter += 1
                silence_counter = 0
                
                # Start recording if not already speaking
                if not is_speaking and speech_counter > 5:
                    is_speaking = True
                    speech_buffer = list(audio_buffer)[-sample_rate:]
                    segment_count += 1
                    print(f"\n🎤 [SEGMENT {segment_count}] Speech detected - Recording...")
                
                # Continue recording
                if is_speaking:
                    speech_buffer.extend(audio_chunk)
                    
            else:
                silence_counter += 1
                speech_counter = max(0, speech_counter - 1)
                
                # Check if speech segment should end
                if is_speaking:
                    speech_buffer.extend(audio_chunk)
                    
                    if silence_counter > int(max_silence_samples / model_frame_size):
                        # End of speech segment
                        is_speaking = False
                        
                        # Check minimum duration
                        if len(speech_buffer) >= min_speech_samples:
                            print(f"\n⏹️  [SEGMENT {segment_count}] Speech ended - Transcribing...")
                            
                            # Transcribe and save
                            text, audio_path, txt_path = transcribe_and_save(speech_buffer, segment_count)
                            
                            duration = len(speech_buffer) / sample_rate
                            print(f"📝 [SEGMENT {segment_count}] Duration: {duration:.2f}s")
                            print(f"📝 [SEGMENT {segment_count}] Transcript: \"{text}\"")
                            if audio_path:
                                print(f"💾 Audio saved: {audio_path}")
                            if txt_path:
                                print(f"💾 Transcript saved: {txt_path}")
                            print("-" * 70)
                            print("🎤 Listening... (speak to transcribe more)")
                        else:
                            print(f"\n⚠️  [SEGMENT {segment_count}] Segment too short, skipped")
                        
                        speech_buffer = []
                        speech_counter = 0
            
            # Display status
            if is_speaking:
                status_bar = "█" * min(int(speech_prob * 20), 20)
                print(f'\r🎤 RECORDING [{status_bar:<20}] {speech_prob:.2f} | Segment {segment_count}   ', end='')
            else:
                status_bar = "░" * 20
                if segment_count > 0:
                    print(f'\r🔇 LISTENING [{status_bar}] | Last: {segment_count} segments   ', end='')
                else:
                    print(f'\r🔇 LISTENING [{status_bar}] | Waiting for speech...   ', end='')
    
    # Start microphone stream
    print("\n🎙️  Real-time Speech Recognition Active")
    print("📊 Speech will be automatically detected and transcribed")
    print("💾 Transcriptions saved to:")
    print(f"   - Individual: {os.path.abspath(OUTPUT_FOLDER)}/segment_XXX.txt")
    print(f"   - Audio: {os.path.abspath(SEGMENTS_FOLDER)}/segment_XXX.wav")
    print("💡 Speak clearly and pause between sentences")
    print("Press Ctrl+C to stop and generate combined files")
    print("-" * 70)
    
    try:
        with sd.InputStream(
            callback=callback,
            channels=1,
            samplerate=sample_rate,
            blocksize=model_frame_size
        ):
            while True:
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("🛑 Recording stopped by user")
        
        # Save combined files
        save_combined_files()
        
        # Print summary
        print("\n" + "=" * 70)
        print("📊 SESSION SUMMARY")
        print("=" * 70)
        print(f"Total segments transcribed: {segment_count}")
        print(f"Successful transcriptions: {len([t for t in all_transcripts if not t.startswith('[Error')])}")
        print(f"Output folders:")
        print(f"  - Transcripts: {os.path.abspath(OUTPUT_FOLDER)}")
        print(f"  - Audio segments: {os.path.abspath(SEGMENTS_FOLDER)}")
        print("=" * 70)
        
        # Print all transcripts
        if all_transcripts:
            print("\n📋 ALL TRANSCRIPTS:")
            print("-" * 70)
            for summary in transcript_summary:
                print(f"[{summary['time']}] Segment {summary['number']:03d}: {summary['text']}")
            print("-" * 70)

if __name__ == "__main__":
    realtime_vad_asr()