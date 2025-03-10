import os
import subprocess

############################
# 1) CONFIGURE PATHS
############################
# Path to SMILExtract binary (adjust this!)
OPENSMILE_PATH = r"C:\Users\USER\Downloads\opensmile-3.0.2-windows-x86_64\opensmile-3.0.2-windows-x86_64\bin\SMILExtract.exe"  

# Path to the IS13 ComParE config:
IS13_CONFIG = r"C:\Users\USER\Downloads\opensmile-3.0.2-windows-x86_64\opensmile-3.0.2-windows-x86_64\config\is09-13\IS13_ComParE.conf"

# Input & output directories
INPUT_DIR = "./AudioSegmentsDenoised/"   # Or "./AudioSegments/"
OUTPUT_DIR = "./AudioFeatures/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_features_for_chunk(chunk_wav, out_csv):
    """
    Calls openSMILE with the IS13-ComParE config on the given chunk_wav.
    Produces a CSV with ~6373 columns describing the audio chunk.
    """

    # 2) BUILD THE COMMAND
    cmd = [
        OPENSMILE_PATH,
        "-C", IS13_CONFIG,      # Use IS13-ComParE config
        "-I", chunk_wav,        # Input WAV chunk
        "-O", out_csv,          # Output CSV
        "-csvoutput", "true" 
    ]

    # 3) RUN THE COMMAND
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def process_all_chunks():
    """
    Iterates over each category (Deceptive/Truthful),
    each video subfolder, and each .wav chunk inside.
    Runs openSMILE to extract features and saves them in ./AudioFeatures/.
    """
    for category in ["Deceptive", "Truthful"]:
        cat_input_dir = os.path.join(INPUT_DIR, category)
        cat_output_dir = os.path.join(OUTPUT_DIR, category)
        os.makedirs(cat_output_dir, exist_ok=True)

        # Each video is a subfolder containing .wav chunks
        for video_name in os.listdir(cat_input_dir):
            video_folder = os.path.join(cat_input_dir, video_name)
            if not os.path.isdir(video_folder):
                continue

            # Create matching output folder for the CSV feature files
            video_out_folder = os.path.join(cat_output_dir, video_name)
            os.makedirs(video_out_folder, exist_ok=True)

            print(f"\n🔎 Extracting openSMILE features for {category}/{video_name} ...")

            # Gather all .wav chunk files
            chunk_files = [f for f in os.listdir(video_folder) if f.endswith(".wav")]
            for chunk_file in chunk_files:
                chunk_path = os.path.join(video_folder, chunk_file)
                # We'll name the CSV similarly to the chunk file but with .csv extension
                base_name = os.path.splitext(chunk_file)[0]  # "chunk_0000"
                out_csv = os.path.join(video_out_folder, f"{base_name}.csv")

                # Extract features for this chunk
                extract_features_for_chunk(chunk_path, out_csv)

    print("\n✅ openSMILE feature extraction complete! Check:", OUTPUT_DIR)


