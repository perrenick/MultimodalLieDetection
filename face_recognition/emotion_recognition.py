from deepface import DeepFace

def analyze_emotion(image_path):
    """
    Returns the dominant emotion for the face in the provided image_path.
    If multiple faces or no faces are detected, the library should handle it,
    but we'll assume only one face per image (since it's cropped).
    """
    # DeepFace.analyze returns a dictionary with emotion details
    result = DeepFace.analyze(img_path=image_path, 
                              actions=['emotion'], 
                              enforce_detection=False)
    return result[0]["dominant_emotion"]  # e.g. "happy"