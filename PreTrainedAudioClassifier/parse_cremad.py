import os
import pandas as pd

from audio_recognition.denoise_audio import denoise_audio_file
CREMAD_DIR = './AudioWAV_CREMAD'
OUTPUT_DIR = './AudioWAV_Denoised'

data_list = []

def denoise_cremad_audio():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for filename in os.listdir(CREMAD_DIR):
        audio_input_path = os.path.join(CREMAD_DIR, filename)
        audio_output_path = os.path.join(OUTPUT_DIR,filename)
        
        print(f"\n🔈 Denoising segments: {filename}")
        denoise_audio_file(audio_input_path,audio_output_path)

def make_dataframe(dir):
    for filename in os.listdir(dir):
        if filename.endswith('.wav'):
            parts = filename.split('_')
            emotion_label = parts[2]

            full_path = os.path.join(OUTPUT_DIR, filename)
            data_list.append([full_path, emotion_label])

    df=pd.DataFrame(data_list, columns=["filepath", "emotion_label"])
    
    return df

