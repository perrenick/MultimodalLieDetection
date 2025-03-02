import cv2 as cv
import os
from tqdm import tqdm
from face_recognition.detect_faces import FaceDetector

# Define input and output directories
INPUT_DIR = "./Clips/"
OUTPUT_DIR = "./ProcessedFaces/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize the face detector
detector = FaceDetector()

# Process each video in Deceptive and Truthful folders
for category in ["Deceptive", "Truthful"]:
    input_path = os.path.join(INPUT_DIR, category)
    output_path = os.path.join(OUTPUT_DIR, category)
    os.makedirs(output_path, exist_ok=True)

    for video_file in os.listdir(input_path):
        if not video_file.endswith(".mp4"):
            continue  # Skip non-video files

        video_path = os.path.join(input_path, video_file)
        video_name = os.path.splitext(video_file)[0]
        video_output_dir = os.path.join(output_path, video_name)
        os.makedirs(video_output_dir, exist_ok=True)

        print(f"\n🔄 Processing: {video_file}...")

        # Open the video
        cap = cv.VideoCapture(video_path)
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))  # Get total number of frames

        frame_id = 0

        # Progress bar for frame processing
        with tqdm(total=total_frames, desc=f"⏳ {video_file}", unit="frame") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break  # End of video

                # Detect and crop the largest face
                face_crop = detector.detect_largest_face(frame)

                if face_crop is not None and face_crop.size > 0:  # ✅ Ensure valid face before saving
                    frame_filename = os.path.join(video_output_dir, f"frame_{frame_id:04d}.jpg")
                    cv.imwrite(frame_filename, face_crop)  # ✅ Save cropped face instead of full frame

                frame_id += 1
                pbar.update(1)  # Update progress bar

        cap.release()

print("\n✅ Processing complete. Faces saved in:", OUTPUT_DIR)
