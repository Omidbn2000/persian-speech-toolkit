import os
import glob
import re
import nemo.collections.asr as nemo_asr
from pathlib import Path
current = Path(__file__).resolve().parent
PROJECT_ROOT = current.parent.parent.parent
# Natural sort function for filenames with numbers
def natural_sort_key(filename):
    """
    Sort filenames naturally (e.g., segment_1, segment_2, segment_10 instead of segment_1, segment_10, segment_2)
    """
    # Extract the base filename without path
    basename = os.path.basename(filename)
    # Split by numbers and convert to int where possible
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split('([0-9]+)', basename)]

MODEL_PATH = PROJECT_ROOT/ "models/Shenava-Koochik-v1.0/shenava-koochik-1.0.nemo"
SEGMENTS_FOLDER = r"C:\Users\sejron\Desktop\Embedded_AI\Phase2\scr\test_scripts\VAD\segments"
OUTPUT_FOLDER = str(current) + "/transcripts"

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("Loading Shenava Koochik v1.0...")

asr_model = nemo_asr.models.ASRModel.restore_from(
    MODEL_PATH
)

asr_model.eval()

print("Model loaded!")
print("Model type:", type(asr_model))

# --------------------------------------------------
# IMPORTANT: Use CTC decoding
# --------------------------------------------------

print("Changing decoder to CTC...")

asr_model.change_decoding_strategy(
    decoder_type="ctc"
)

# --------------------------------------------------
# IMPORTANT: Set Shenava's recommended context
# --------------------------------------------------

print("Setting attention context to [70, 13]...")

asr_model.encoder.set_default_att_context_size(
    [70, 13]
)

print("Decoder:", asr_model.decoding)

# --------------------------------------------------
# Find audio files and sort naturally
# --------------------------------------------------

segment_files = glob.glob(os.path.join(SEGMENTS_FOLDER, "*.wav"))

# Sort files naturally by numeric order
segment_files = sorted(segment_files, key=natural_sort_key)

print(f"\nFound {len(segment_files)} files")
print("Files in order:")
for i, file_path in enumerate(segment_files, 1):
    print(f"  {i}. {os.path.basename(file_path)}")

# --------------------------------------------------
# Initialize list to store all transcripts
# --------------------------------------------------

all_transcripts = []
transcript_summary = []

# --------------------------------------------------
# Transcribe and save to TXT files
# --------------------------------------------------

for i, file_path in enumerate(segment_files, 1):

    print("\n" + "=" * 80)
    print(f"Segment {i}: {os.path.basename(file_path)}")
    print("=" * 80)

    try:

        results = asr_model.transcribe(
            [file_path],
            batch_size=1
        )

        result = results[0]

        if hasattr(result, "text"):
            text = result.text
        else:
            text = str(result)

        # Clean up empty transcripts
        if text.strip() == "":
            text = "[EMPTY TRANSCRIPT]"
            print("\nWARNING: Empty transcript detected!")

        print("\nTRANSCRIPTION:")
        print(text)

        # Store transcript in list
        all_transcripts.append(text)
        
        # Add file info and transcript to summary (with padded number for better sorting)
        file_info = os.path.basename(file_path)
        transcript_summary.append(f"--- {file_info} ---\n{text}\n")

        # Save individual transcription to TXT file
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        txt_filename = f"{base_filename}.txt"
        txt_path = os.path.join(OUTPUT_FOLDER, txt_filename)
        
        # Write transcription to file
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"\nTranscription saved to: {txt_path}")

    except Exception as e:

        print("\nERROR:")
        print(repr(e))
        
        # Save error message to TXT file
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        txt_filename = f"{base_filename}_error.txt"
        txt_path = os.path.join(OUTPUT_FOLDER, txt_filename)
        
        error_msg = f"ERROR: {repr(e)}"
        all_transcripts.append(error_msg)
        
        file_info = os.path.basename(file_path)
        transcript_summary.append(f"--- {file_info} ---\n{error_msg}\n")
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(error_msg)
        
        print(f"Error logged to: {txt_path}")

# --------------------------------------------------
# Save all transcripts to a single file (in correct order)
# --------------------------------------------------

print("\n" + "=" * 80)
print("Generating combined transcript file...")
print("=" * 80)

# Create a summary file with all transcripts in order
summary_filename = "all_transcripts.txt"
summary_path = os.path.join(OUTPUT_FOLDER, summary_filename)

with open(summary_path, 'w', encoding='utf-8') as f:
    # Write header
    f.write("=" * 80 + "\n")
    f.write("ALL TRANSCRIPTS - COMBINED FILE\n")
    f.write(f"Total segments: {len(segment_files)}\n")
    f.write(f"Generated on: {os.path.basename(OUTPUT_FOLDER)}\n")
    f.write("=" * 80 + "\n\n")
    
    # Write all transcripts with separators (in correct order)
    for summary in transcript_summary:
        f.write(summary)
        f.write("\n" + "-" * 80 + "\n\n")
    
    # Write footer
    f.write("=" * 80 + "\n")
    f.write("END OF TRANSCRIPTS\n")
    f.write("=" * 80 + "\n")

print(f"\nCombined transcripts saved to: {summary_path}")

# Also create a plain version with just the text (no separators)
plain_summary = "all_transcripts_plain.txt"
plain_path = os.path.join(OUTPUT_FOLDER, plain_summary)

with open(plain_path, 'w', encoding='utf-8') as f:
    for text in all_transcripts:
        f.write(text + "\n\n")

print(f"Plain combined transcripts saved to: {plain_path}")

# --------------------------------------------------
# Also create a continuous text version (all transcripts concatenated)
# --------------------------------------------------

continuous_summary = "all_transcripts_continuous.txt"
continuous_path = os.path.join(OUTPUT_FOLDER, continuous_summary)

with open(continuous_path, 'w', encoding='utf-8') as f:
    for text in all_transcripts:
        f.write(text + " ")

print(f"Continuous transcripts saved to: {continuous_path}")

# --------------------------------------------------
# Print summary statistics
# --------------------------------------------------

print("\n" + "=" * 80)
print("TRANSCRIPTION COMPLETE!")
print("=" * 80)
print(f"Total segments processed: {len(segment_files)}")
print(f"All transcripts saved to: {OUTPUT_FOLDER}")
print(f"Combined file (with separators): {summary_filename}")
print(f"Combined file (plain text): {plain_summary}")
print(f"Combined file (continuous): {continuous_summary}")
print("=" * 80)

# Print the transcript order to verify
print("\nTranscripts in order:")
for i, (file_path, text) in enumerate(zip(segment_files, all_transcripts), 1):
    preview = text[:50] + "..." if len(text) > 50 else text
    print(f"{i}. {os.path.basename(file_path)}: {preview}")