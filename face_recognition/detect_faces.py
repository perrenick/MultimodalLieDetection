import cv2 as cv
import os
import torch
import numpy as np
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt
from math import sqrt
import math 
from deepface import DeepFace

class FaceDetector():
    def __init__(self, keep_all=True, device=None, output_folder='./Processed_Faces', save_faces=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(keep_all=keep_all, device=self.device)
        self.save_faces = save_faces
        self.output_folder = output_folder

        os.makedirs(output_folder, exist_ok=True)
        self.target_box = None
        self.frames_missing = 0
        self.missing_thresh = 5
        self.target_found = False

    def people_finder(self, frame):
        boxes, _ = self.mtcnn.detect(frame)

        if boxes is None:  # No face detected
            return False  

        if len(boxes) > 1:  # More than one face detected
            return True  

        return False  # Default case: One or no face detected

        
    def detect_faces(self, frame, frame_id, ref_img_path):
        
        # if self.is_blurry(frame):
        #     print(f"Skipping frame {frame_id}: Blurry")
        #     return None
    
        boxes, _, landmarks = self.mtcnn.detect(frame, landmarks=True)


        if boxes is not None:
            if frame_id==1: #First frame that i have to find the target box
                for i,box in enumerate(boxes): #makes boxes iterable
                    # if self.is_too_small(box):
                    #     print(f"Skipping frame {frame_id}: Face too small")
                    #     continue
                    
                    x, y, x2, y2 = map(int, box)

                    detected_face = frame[int(y):int(y2), int(x):int(x2)]
                    temp_face_path = f"temp_face_{i}.jpg"
                    cv.imwrite(temp_face_path, detected_face)

                    try:
                        result = DeepFace.verify(temp_face_path, ref_img_path, model_name="Facenet", enforce_detection=False)
                        verified = result['verified']

                        if verified:
                            print(f"Face {i} is the correct one!")
                            self.target_box = (x, y, x2, y2)  # Save the target face's bounding box
                            self.target_found = True
                            break  # Stop checking after finding the correct face

                    except Exception as e:
                        print(f"Error comparing face {i}: {str(e)}")
            else:
                if self.target_box:
                    prev_x, prev_y, prev_x2, prev_y2 = self.target_box
                    prev_center = ((prev_x + prev_x2) // 2, (prev_y + prev_y2) // 2)

                    best_match = None
                    min_distance = float("inf")

                    for box in boxes:
                        x, y, x2, y2 = map(int, box)
                        face_center = ((x + x2) // 2, (y + y2) // 2)
                        distance = np.linalg.norm(np.array(prev_center) - np.array(face_center))

                        if distance < min_distance:
                            min_distance = distance
                            best_match = (x, y, x2, y2)

                    if best_match:
                        self.target_box = best_match  # Update tracking box

                if self.target_box:
                    x, y, x2, y2 = self.target_box
                    cv.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 3)
                # rotated_frame = self.align_face(frame, landmarks[0][0], landmarks[0][1])

                # face = rotated_frame[int(y):int(y2), int(x):int(x2)] #extract (crop) only the face

                # if self.save_faces:
                #     face_filename = os.path.join(self.output_folder, f"frame_{frame_id}_face_{i}.jpg")
                #     cv.imwrite(face_filename, face)

                # cv.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 5)
        
        if landmarks is not None:
            for face_landmarks in landmarks:
                for x,y in face_landmarks:
                    cv.circle(frame, (int(x),int(y)), 3, (255, 0, 0))

        return frame


