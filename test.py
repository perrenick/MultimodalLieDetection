# import cv2 as cv
# import os
# import torch
# import numpy as np
# from facenet_pytorch import MTCNN
# import matplotlib.pyplot as plt
# from math import sqrt
# import math 

# def match_face(detected_face, ref_image):
#         detected_face = cv.cvtColor(detected_face, cv.COLOR_BGR2GRAY)
#         detected_face = cv.resize(detected_face, (300, 300))  # Normalize size
        
        
#         mse = np.mean((ref_image - detected_face) ** 2)
        
#         return mse < 500  # Threshold (μπορεί να ρυθμιστεί)

# ref_img_path='./trial_056.jpg'
# cap = cv.VideoCapture('./Clips/Deceptive/trial_lie_056.mp4')

# ret,frame = cap.read()

# ref_image = cv.imread(ref_img_path)

# ref_image = cv.cvtColor(ref_image, cv.COLOR_BGR2GRAY)
# ref_image = cv.resize(ref_image, (300, 300))  # Normalize size
# mtcnn = MTCNN()

# boxes, _ = mtcnn.detect(frame)

# for i, box in enumerate(boxes):
#     x,y,x2,y2 = map(int, box)

#     cv.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 5)

#     detected_face = frame[int(y):int(y2), int(x):int(x2)]

#     result = match_face(detected_face, ref_image)
#     print(result)
# cv.imshow('frame', frame)
# cv.imshow('ref_image',ref_image)
# cv.waitKey(0)
import cv2 as cv
import numpy as np
from facenet_pytorch import MTCNN
from deepface import DeepFace

def is_too_small(box, min_size=30):
    """Returns True if the face is too small based on width and height."""
    x, y, x2, y2 = map(int, box)
    return (x2 - x) < min_size or (y2 - y) < min_size  # Width or height too small
# Initialize MTCNN
def is_blurry(face_crop, threshold=50):
    """Returns True if the face is blurry based on Laplacian variance."""
    if face_crop is None or face_crop.size == 0:
        return True  # Avoid errors when no face is detected

    gray = cv.cvtColor(face_crop, cv.COLOR_BGR2GRAY)
    variance = cv.Laplacian(gray, cv.CV_64F).var()
    
    print(f"Blurriness Score: {variance}")  # Debugging to see blur values

    return variance < threshold  # If variance is low, it's blurry

mtcnn = MTCNN(keep_all=True)

# Load Reference Face
ref_img_path = './trial_056.jpg'
ref_image = cv.imread(ref_img_path)

# Open Video and Read First Frame
cap = cv.VideoCapture('./Clips/Deceptive/trial_lie_035.mp4')


# Detect Faces in the First Frame
major_frame_size_x = 0
major_frame_size_y = 0
frame_id=0
while True:
    ret, frame = cap.read()

    if not ret:
        print(f'End of video')
        break

    boxes, probs, landmarks= mtcnn.detect(frame, landmarks=True)

   
    if boxes is not None:
        face_areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in boxes]
        max_face_area = max(face_areas) if face_areas else 0
        for i, box in enumerate(boxes):
            
            # **Step 1: Find the largest face (by area)**
            
            x1, y1, x2, y2 = map(int, box)
            face_area = (x2-x1)*(y2-y1)

            if probs[i]<0.99:
                cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 5)
                continue

            if face_area < 0.8 * max_face_area:  # Ignore faces smaller than 50% of the largest
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)  # Green for background faces
                continue 

            # if is_blurry(frame[int(y1):int(y2), int(x1):int(x2)]):
            #     print(f"Face {i} is blurry!")  # Debugging
            #     cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 5)  # Magenta for blurry faces
            #     continue  # Skip blurry faces
            
            if(x2 - x1) < 30 or (y2 - y1) < 30:
                cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 5)
            # elif landmarks is None or len(landmarks) == 0 or landmarks[i] is None:
            #     cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)  # Green for missing landmarks
            else:
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
            
    cv.imshow('Frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit early
        break
cap.release()
cv.destroyAllWindows()
#     x, y, x2, y2 = map(int, box)
#     detected_face = frame[int(y):int(y2), int(x):int(x2)]  # Extract face

#     # Save face temporarily for DeepFace comparison
#     temp_face_path = f"temp_face_{i}.jpg"
#     cv.imwrite(temp_face_path, detected_face)

#     # Compare with Reference Face using DeepFace
#     try:
#         result = DeepFace.verify(temp_face_path, ref_img_path, model_name="Facenet", enforce_detection=False)
#         distance = result['distance']
#         threshold = result['threshold']
#         verified = result['verified']

#         if verified:
#             print(f"Face {i} matches with reference! Distance: {distance:.2f}")
#             cv.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 5)  # Green for match
#         else:
#             print(f"Face {i} does NOT match. Distance: {distance:.2f}")
#             cv.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 5)  # Red for non-match

#     except Exception as e:
#         print(f"Error comparing face {i}: {str(e)}")

# # Show Results
# cv.imshow('frame', frame)
# cv.imshow('ref_image', ref_image)
# cv.waitKey(0)
# cap.release()
# cv.destroyAllWindows()

# def align_face(self, face, left_eye, right_eye):
#         a = abs(left_eye[1] - right_eye[1])  # Vertical difference
#         b = abs(left_eye[0] - right_eye[0])  # Horizontal difference

#         c = sqrt(a * a + b * b)  # Euclidean distance (not actually needed for rotation)

#         # Compute the rotation angle (using arctan)
#         alpha = np.arctan2(a, b)  # Angle in radians
#         alpha = (alpha * 180) / math.pi  # Convert to degrees

#         # Get the center between the eyes
#         eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

#         # Rotation matrix
#         h, w = face.shape[:2]
#         rotation_matrix = cv.getRotationMatrix2D(eye_center, alpha, 1)

#         # Rotate the face
#         aligned_face = cv.warpAffine(face, rotation_matrix, (w, h), flags=cv.INTER_CUBIC)

#         return aligned_face
