import os
import pandas as pd
from tqdm import tqdm
# Root folder where openSMILE outputs live, e.g. ./AudioFeatures/Deceptive/trial_lie_001/chunk_0000.csv
AUDIO_FEATURES_DIR = "./AudioFeatures/"
OUTPUT_CSV = "merged_audio_features.csv"

def parse_opensmile_arff(csv_file):
    """
    Reads a single openSMILE ARFF-like CSV and returns a list of floats
    (the numeric features) plus the first token (e.g. "unknown").
    """
    with open(csv_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Find '@data' line
    data_index = None
    for i, line in enumerate(lines):
        if "@data" in line:
            data_index = i
            break
    
    if data_index is None:
        return None, None  # No @data found
    
    # **Find the first non-empty line after @data**
    for i in range(data_index + 1, len(lines)):
        data_line = lines[i].strip()
        if data_line:  # If not empty
            break
    else:
        return None, None  # No valid data found
    
    # ✅ Now proceed safely
    values = data_line.split(",")  
    name_field = values[0].replace("'", "").strip()  # Remove extra quotes

    # ✅ Handle missing values ("?") by replacing them with None
    numeric_features = []
    for v in values[1:]:
        v = v.strip()
        if v == "?":
            numeric_features.append(None)  # Replace "?" with None
        else:
            try:
                numeric_features.append(float(v))  # Convert to float
            except ValueError:
                numeric_features.append(None)  # Handle unexpected values
    
    return name_field, numeric_features



def merge_all_chunks():
    """
    Loops through AUDIO_FEATURES_DIR (Deceptive/Truthful), 
    merges all chunk_{xxxx}.csv files into one DataFrame, and saves as CSV.
    """
    rows = []
    total_files = sum(len(files) for _, _, files in os.walk(AUDIO_FEATURES_DIR))  # Count all CSV files
    
    with tqdm(total=total_files, desc="🔄 Merging Audio Chunks", unit="file") as pbar:
        for category in ["Deceptive", "Truthful"]:
            cat_path = os.path.join(AUDIO_FEATURES_DIR, category)
            if not os.path.isdir(cat_path):
                continue
            
            for video_name in os.listdir(cat_path):
                video_folder = os.path.join(cat_path, video_name)
                if not os.path.isdir(video_folder):
                    continue
                
                # Gather all chunk files
                chunk_files = [f for f in os.listdir(video_folder) if f.endswith(".csv")]
                
                for chunk_file in chunk_files:
                    chunk_path = os.path.join(video_folder, chunk_file)
                    
                    # e.g. "chunk_0000.csv" -> chunk_id = "chunk_0000"
                    chunk_id = os.path.splitext(chunk_file)[0]
                    
                    name_field, features = parse_opensmile_arff(chunk_path)
                    if features is None:
                        # skip invalid
                        continue
                    
                    # Build a dict for this row
                    row_dict = {
                        "video_name": video_name,
                        "chunk_id": chunk_id,
                        "name_field": name_field,  # e.g. 'unknown'
                    }
                    
                    # Add numeric features with columns like f0, f1, ...
                    # or keep them as a list until we define columns
                    for i, val in enumerate(features):
                        row_dict[f"feat_{i}"] = val
                    
                    rows.append(row_dict)
            
    
    # Convert the list of dicts to a DataFrame
    df = pd.DataFrame(rows)
    
    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Merged {len(df)} rows into {OUTPUT_CSV}")



