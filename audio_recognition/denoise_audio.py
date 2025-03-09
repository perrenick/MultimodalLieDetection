import os
import librosa
import soundfile as sf
import noisereduce as nr

INPUT_DIR = "./AudioSegments/"
OUTPUT_DIR = "./AudioSegmentsDenoised/"

def denoise_audio_file(input_path, output_path):
    """
    Loads a WAV file, performs noise reduction, and saves the result.
    """
    # Load the file at its native sample rate (sr=None).
    y, sr = librosa.load(input_path, sr=None)

    # Apply noise reduction.
    # noisereduce automatically estimates noise from the audio signal.
    reduced_noise = nr.reduce_noise(y=y, sr=sr)

    # Save the denoised signal back to a WAV file
    sf.write(output_path, reduced_noise, sr)

def denoise_all_segments():
    """
    Loops over each 0.5s chunk in AudioSegments/,
    denoises them, and saves to AudioSegmentsDenoised/.
    """
    # Create output directory if needed
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for category in ["Deceptive", "Truthful"]:
        category_input = os.path.join(INPUT_DIR, category)
        category_output = os.path.join(OUTPUT_DIR, category)
        os.makedirs(category_output, exist_ok=True)

        # Each subfolder corresponds to a video name
        for video_name in os.listdir(category_input):
            video_input_path = os.path.join(category_input, video_name)
            video_output_path = os.path.join(category_output, video_name)

            if not os.path.isdir(video_input_path):
                continue  # Skip if it's not a folder

            os.makedirs(video_output_path, exist_ok=True)

            print(f"\n🔈 Denoising segments: {category}/{video_name}")

            # Gather all .wav chunks
            chunk_files = [f for f in os.listdir(video_input_path) if f.endswith(".wav")]
            for chunk_file in chunk_files:
                input_path = os.path.join(video_input_path, chunk_file)
                output_path = os.path.join(video_output_path, chunk_file)

                denoise_audio_file(input_path, output_path)

    print("\n✅ All chunks denoised and saved in", OUTPUT_DIR)
