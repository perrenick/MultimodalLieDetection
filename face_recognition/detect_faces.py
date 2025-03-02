import cv2 as cv
import numpy as np
from facenet_pytorch import MTCNN

class FaceDetector:
    def __init__(self, min_confidence=0.99, min_size=30):
        """Initialize MTCNN for face detection."""
        self.mtcnn = MTCNN(keep_all=True)
        self.min_confidence = min_confidence
        self.min_size = min_size

    def is_too_small(self, box):
        """Check if a detected face is too small."""
        x1, y1, x2, y2 = map(int, box)
        return (x2 - x1) < self.min_size or (y2 - y1) < self.min_size

    def detect_largest_face(self, frame):
        """
        Detect the largest face in the frame and return its cropped image.
        Returns:
        - face_crop (numpy array): Cropped face image if found, else None.
        """
        # Handle cases where only (boxes, probs) are returned
        detection_result = self.mtcnn.detect(frame)
        if len(detection_result) == 2:
            boxes, probs = detection_result
            landmarks = None  # Landmarks are missing
        else:
            boxes, probs, landmarks = detection_result  # Normal case

        if boxes is None:
            return None  # No faces detected

        # Find the largest face
        face_areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in boxes]
        max_index = np.argmax(face_areas)
        x1, y1, x2, y2 = map(int, boxes[max_index])  # Get bounding box

        # Apply filters
        if probs[max_index] < self.min_confidence or self.is_too_small(boxes[max_index]):
            return None  # Ignore low-confidence or too small faces

        # Ensure valid crop dimensions
        if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
            return None  # Prevent cropping outside image boundaries

        # Crop the detected face
        face_crop = frame[y1:y2, x1:x2].copy()  # Ensure a proper copy

        return face_crop  # Return the cropped face image
