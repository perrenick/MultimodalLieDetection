from PreTrainedAudioClassifier.parse_cremad import make_dataframe
import os
import subprocess

OPENSMILE_PATH = r"C:\Users\USER\Downloads\opensmile-3.0.2-windows-x86_64\opensmile-3.0.2-windows-x86_64\bin\SMILExtract.exe"  

# Path to the IS13 ComParE config:
IS13_CONFIG = r"C:\Users\USER\Downloads\opensmile-3.0.2-windows-x86_64\opensmile-3.0.2-windows-x86_64\config\is09-13\IS13_ComParE.conf"
OUTPUT_DIR = './AudioWAV_Denoised'
OUTPUT_OPENSMILE_DIR = './AudioWAV_Features'
INPUT_DIR = './AudioWAV_Denoised'

def extract_features_for_crema():

    df = make_dataframe(OUTPUT_DIR)

    for idx, row in df.iterrows():
        wav_path = row["filepath"]
        out_csv = os.path.join(OUTPUT_OPENSMILE_DIR, f"{idx}.csv")
        print(f"\n🔎 Extracting openSMILE features for {wav_path} ...")
        
        cmd = [
            OPENSMILE_PATH,
            "-C", IS13_CONFIG, 
            "-I", wav_path,
            "-O", out_csv
        ]
        subprocess.run(cmd)

def process_all_videos_crema():

    os.makedirs(OUTPUT_OPENSMILE_DIR, exist_ok=True)
    
    extract_features_for_crema()

    print("\n✅ openSMILE feature extraction complete! Check:", OUTPUT_DIR)

