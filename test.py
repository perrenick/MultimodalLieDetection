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

import os
import librosa
import soundfile as sf
import noisereduce as nr

input_path = './Audio/Deceptive/trial_lie_007.wav'
output_path = './Audio_Test'
y, sr = librosa.load(input_path, sr=None)

os.makedirs(output_path, exist_ok=True)
# Apply noise reduction.
# noisereduce automatically estimates noise from the audio signal.
reduced_noise = nr.reduce_noise(y=y, sr=sr)

output_filename = os.path.join(output_path, 'trial_lie_007_denoised.wav')

# Write to disk
sf.write(output_filename, reduced_noise, sr)
