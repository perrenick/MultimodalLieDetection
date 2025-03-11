# while True:
#     ret, frame = cap.read()

#     if not ret:
#         print(f'End of video')
#         break

#     boxes, probs, landmarks= mtcnn.detect(frame, landmarks=True)

   
#     if boxes is not None:
#         face_areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in boxes]
#         max_face_area = max(face_areas) if face_areas else 0
#         for i, box in enumerate(boxes):
            
#             # **Step 1: Find the largest face (by area)**
            
#             x1, y1, x2, y2 = map(int, box)
#             face_area = (x2-x1)*(y2-y1)

#             if probs[i]<0.99:
#                 cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 5)
#                 continue

#             if face_area < 0.8 * max_face_area:  # Ignore faces smaller than 50% of the largest
#                 cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)  # Green for background faces
#                 continue 


# from deepface import DeepFace
# import cv2 as cv

# image = cv.imread('ProcessedFaces/Deceptive/trial_lie_004/frame_0000.jpg')

# predictions = DeepFace.analyze(image)

# print(predictions)

# import pandas as pd

# dir = './Annotation/All_Gestures_Deceptive and Truthful.csv'

# df = pd.read_csv(dir)

# print(df.head())

# import os
# import librosa
# import soundfile as sf
# import noisereduce as nr

# input_path = './Audio/Deceptive/trial_lie_007.wav'
# output_path = './Audio_Test'
# y, sr = librosa.load(input_path, sr=None)

# os.makedirs(output_path, exist_ok=True)
# # Apply noise reduction.
# # noisereduce automatically estimates noise from the audio signal.
# reduced_noise = nr.reduce_noise(y=y, sr=sr)

# output_filename = os.path.join(output_path, 'trial_lie_007_denoised.wav')

# # Write to disk
# sf.write(output_filename, reduced_noise, sr)

import pandas as pd

# Path to the CSV file
# csv_file = "./AudioFeatures/Deceptive/trial_lie_001/chunk_0000.csv"

# Find where @data starts
# data_start = next(i for i, line in enumerate(lines) if "@data" in line) + 1

# # Read only the data part into a DataFrame
# df = pd.read_csv(csv_file, skiprows=data_start, header=None, sep=",")

# # Print first rows of the numerical data
# print(df.columns())

import os
import pandas as pd
from PreTrainedAudioClassifier.parse_cremad import make_dataframe

# from audio_recognition.denoise_audio import denoise_audio_file
# CREMAD_DIR = './AudioWAV_CREMAD'
# OUTPUT_DIR = './AudioWAV_Denoised'

# data_list = []

# def denoise_cremad_audio():

#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     for filename in os.listdir(CREMAD_DIR):
#         audio_input_path = os.path.join(CREMAD_DIR, filename)
#         audio_output_path = os.path.join(OUTPUT_DIR,filename)
        
#         print(f"\n🔈 Denoising segments: {filename}")
#         denoise_audio_file(audio_input_path,audio_output_path)

# # denoise_cremad_audio()

# for filename in os.listdir(OUTPUT_DIR):
#     if filename.endswith('.wav'):
#         parts = filename.split('_')
#         emotion_label = parts[2]

#         full_path = os.path.join(OUTPUT_DIR, filename)
#         data_list.append([full_path, emotion_label])

# df = pd.DataFrame(data_list, columns=["filepath", "emotion_label"])
# print(df.head())

df_labels = make_dataframe('./AudioWAV_Denoised')
df_features = pd.read_csv('merged_audio_wav_features.csv')

# 1. Αφαίρεση της στήλης "name_field" που περιέχει "unknown"
df_features.drop(columns=["name_field"], inplace=True)

# 2. Αφαίρεση των στηλών που περιέχουν NaN τιμές
df_features.dropna(axis=1, inplace=True)

# 3. Δημιουργία mapping για να μετατρέψουμε το "video_name" ώστε να ταιριάζει με το "filename"
df_labels["filename"] = df_labels["filepath"].apply(lambda x: x.split("/")[-1].replace(".wav", ".csv"))

# Αντιστοίχιση των πρώτων N video_names στα ονόματα αρχείων των labels
mapping = {str(i) + ".csv": filename for i, filename in enumerate(df_labels["filename"])}

# Αντικατάσταση των ονομάτων στο df_features σύμφωνα με το mapping
df_features["video_name"] = df_features["video_name"].map(mapping)

# 4. Συγχώνευση των δύο DataFrames
df_merged = df_features.merge(df_labels, left_on="video_name", right_on="filename", how="inner")

# 5. Αφαίρεση της στήλης "filename" αφού δεν χρειάζεται πλέον
df_merged.drop(columns=["filename"], inplace=True)

# 6. Αποθήκευση του τελικού DataFrame (αν χρειάζεται)
df_merged.to_csv("merged_data.csv", index=False)

# Προβολή των πρώτων γραμμών του τελικού DataFrame
print(df_merged.head())

