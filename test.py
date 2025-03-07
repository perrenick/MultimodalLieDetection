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

from pydub import AudioSegment
from pydub.playback import play

path = './Audio/Deceptive/trial_lie_001.wav'
try:
    wav_file = AudioSegment.from_wav(path)
    print("Το αρχείο φορτώθηκε με επιτυχία!")
except Exception as e:
    print(f"Σφάλμα κατά τη φόρτωση του αρχείου: {e}")

play(wav_file)
