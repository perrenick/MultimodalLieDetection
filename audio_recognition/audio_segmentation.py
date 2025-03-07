import os
import math
from pydub import AudioSegment

# Define input/output directories
INPUT_DIR = "./Audio/"
OUTPUT_DIR = "./AudioSegments/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set chunk length (0.5 seconds = 500 milliseconds)
CHUNK_LENGTH_MS = 500

def segment_audio(wav_path, output_dir):
    """
    Splits a WAV file into fixed-length chunks (0.5 seconds).
    Saves them in output_dir.
    """
    audio = AudioSegment.from_wav(wav_path)
    total_ms = len(audio)
    num_chunks = math.ceil(total_ms / CHUNK_LENGTH_MS)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_chunks):
        start = i * CHUNK_LENGTH_MS
        end = start + CHUNK_LENGTH_MS
        chunk = audio[start:end]
        chunk.export(os.path.join(output_dir, f"chunk_{i:04d}.wav"), format="wav")

def process_all_audio():
    """
    Loops through all WAV files in `Audio/`, segments them, and saves chunks.
    """
    for category in ["Deceptive", "Truthful"]:
        input_path = os.path.join(INPUT_DIR, category)
        output_path = os.path.join(OUTPUT_DIR, category)
        os.makedirs(output_path, exist_ok=True)

        for audio_file in os.listdir(input_path):
            if not audio_file.endswith(".wav"):
                continue  # Skip non-audio files

            audio_path = os.path.join(input_path, audio_file)
            video_name = os.path.splitext(audio_file)[0]  # Remove .wav extension
            video_output_dir = os.path.join(output_path, video_name)
            os.makedirs(video_output_dir, exist_ok=True)

            print(f"🔪 Segmenting audio: {audio_file} ...")
            segment_audio(audio_path, video_output_dir)
            print(f"✅ Saved segments in: {video_output_dir}")

