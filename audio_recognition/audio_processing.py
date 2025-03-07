import os
import subprocess

# Define directories
INPUT_DIR = "./Clips/"
OUTPUT_DIR = "./Audio/"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_audio_from_video(video_path, output_wav):
    """
    Extracts audio from a given video file and saves it as a WAV file.
    """
    cmd = [
        "ffmpeg", "-y",              # Overwrite existing files
        "-i", video_path,            # Input video file
        "-ac", "1",                   # Convert to mono
        "-ar", "16000",               # Set sample rate to 16kHz
        "-vn",                        # Disable video output
        output_wav                    # Output audio file
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def process_all_videos():
    """
    Loops through all videos in `Clips/`, extracts audio, and saves as WAV.
    """
    for category in ["Deceptive", "Truthful"]:
        input_path = os.path.join(INPUT_DIR, category)
        output_path = os.path.join(OUTPUT_DIR, category)
        os.makedirs(output_path, exist_ok=True)

        for video_file in os.listdir(input_path):
            if not video_file.endswith(".mp4"):
                continue  # Skip non-video files
            
            video_path = os.path.join(input_path, video_file)
            video_name = os.path.splitext(video_file)[0]  # Remove .mp4 extension
            output_wav = os.path.join(output_path, f"{video_name}.wav")

            print(f"🎵 Extracting audio from: {video_file} ...")
            extract_audio_from_video(video_path, output_wav)
            print(f"✅ Saved: {output_wav}")
