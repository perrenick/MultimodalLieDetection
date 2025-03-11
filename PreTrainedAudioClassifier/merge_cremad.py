from audio_recognition.merge_chunks import parse_opensmile_arff
from tqdm import tqdm
import os
import pandas as pd

AUDIO_FEATURES_DIR = "./AudioWAV_Features/"
OUTPUT_CSV = "merged_audio_wav_features.csv"

def merge_cremad():
    rows = []
    
    file_names = os.listdir(AUDIO_FEATURES_DIR)
    sorted_files = sorted(file_names, key=lambda x: int(x.split(".")[0]))  # Sort numerically

    total_files = len(sorted_files)  # Get total count for progress bar


    with tqdm(total=total_files, desc="🔄 Merging Audio", unit="file") as pbar:
            for video_name in sorted_files:
                video_path = os.path.join(AUDIO_FEATURES_DIR, video_name)
                name_field, features = parse_opensmile_arff(video_path)
                if features is None:
                    # skip invalid
                    continue
                
                # Build a dict for this row
                row_dict = {
                    "video_name": video_name,
                    "name_field": name_field,  # e.g. 'unknown'
                }
                
                # Add numeric features with columns like f0, f1, ...
                # or keep them as a list until we define columns
                for i, val in enumerate(features):
                    row_dict[f"feat_{i}"] = val
                
                rows.append(row_dict)
                pbar.update(1)
    # Convert the list of dicts to a DataFrame
    df = pd.DataFrame(rows)

    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"✅ Merged {len(df)} rows into {OUTPUT_CSV}")