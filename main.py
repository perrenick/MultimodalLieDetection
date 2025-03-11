import cv2 as cv
import os
from tqdm import tqdm
import csv
from face_recognition.detect_faces import FaceDetector
from face_recognition.emotion_recognition import analyze_emotion
from audio_recognition.audio_processing import process_all_videos
from audio_recognition.audio_segmentation import process_all_audio
from audio_recognition.denoise_audio import denoise_all_segments
from audio_recognition.extract_opensmile import process_all_chunks
from audio_recognition.merge_chunks import merge_all_chunks
from PreTrainedAudioClassifier.cremad_opensmile import process_all_videos_crema
from PreTrainedAudioClassifier.merge_cremad import merge_cremad

# Define input and output directories
INPUT_DIR = "./Clips/"
OUTPUT_DIR = "./ProcessedFaces/"
INPUT_DIR_EMOTION = "./ProcessedFaces/"
OUTPUT_CSV = "emotion_predictions.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Initialize the face detector
# detector = FaceDetector()

# # Process each video in Deceptive and Truthful folders
# for category in ["Deceptive", "Truthful"]:
#     input_path = os.path.join(INPUT_DIR, category)
#     output_path = os.path.join(OUTPUT_DIR, category)
#     os.makedirs(output_path, exist_ok=True)

#     for video_file in os.listdir(input_path):
#         if not video_file.endswith(".mp4"):
#             continue  # Skip non-video files

#         video_path = os.path.join(input_path, video_file)
#         video_name = os.path.splitext(video_file)[0]
#         video_output_dir = os.path.join(output_path, video_name)
#         os.makedirs(video_output_dir, exist_ok=True)

#         print(f"\n🔄 Processing: {video_file}...")

#         # Open the video
#         cap = cv.VideoCapture(video_path)
#         total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))  # Get total number of frames

#         frame_id = 0

#         # Progress bar for frame processing
#         with tqdm(total=total_frames, desc=f"⏳ {video_file}", unit="frame") as pbar:
#             while cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret:
#                     break  # End of video

#                 # Detect and crop the largest face
#                 face_crop = detector.detect_largest_face(frame)

#                 if face_crop is not None and face_crop.size > 0:  # ✅ Ensure valid face before saving
#                     frame_filename = os.path.join(video_output_dir, f"frame_{frame_id:04d}.jpg")
#                     cv.imwrite(frame_filename, face_crop)  # ✅ Save cropped face instead of full frame

#                 frame_id += 1
#                 pbar.update(1)  # Update progress bar

#         cap.release()

# print("\n✅ Processing complete. Faces saved in:", OUTPUT_DIR)

# with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as f:
#     writer = csv.writer(f)
#     writer.writerow(["category", "video_name", "frame_id", "emotion"])

#     # We have two categories: "Deceptive" and "Truthful"
#     for category in ["Deceptive", "Truthful"]:
#         category_path = os.path.join(INPUT_DIR_EMOTION, category)
#         if not os.path.isdir(category_path):
#             continue

#         for video_name in os.listdir(category_path):
#             video_folder = os.path.join(category_path, video_name)
#             if not os.path.isdir(video_folder):
#                 continue

#             # Gather all .jpg frames
#             frames = sorted([f for f in os.listdir(video_folder) if f.endswith(".jpg")])

#             # Initialize a tqdm progress bar for this video
#             total_frames = len(frames)
#             print(f"\n🔄 Processing video: {category}/{video_name} ...")
#             with tqdm(total=total_frames, desc=f"⏳ {video_name}", unit="frame") as pbar:
#                 for frame_file in frames:
#                     # path to the face image
#                     frame_path = os.path.join(video_folder, frame_file)
#                     frame_id_str = os.path.splitext(frame_file)[0]  # "frame_0000"

#                     # 1. Load the face image (if needed)
#                     # frame_img = cv.imread(frame_path)

#                     # 2. Run emotion analysis
#                     emotion = analyze_emotion(frame_path)

#                     # 3. Write result to CSV
#                     writer.writerow([category, video_name, frame_id_str, emotion])

#                     # Update the progress bar
#                     pbar.update(1)

# print(f"\n✅ Emotion recognition complete. Results saved in {OUTPUT_CSV}")

# process_all_videos()
# print("\n🎉 Audio extraction complete. WAV files saved in ./Audio/")

# process_all_audio()
# print("\n🎉 Audio segmentation complete! Chunks saved in ./AudioSegments/")

# denoise_all_segments()

# process_all_chunks()

# merge_all_chunks()

# process_all_videos_crema()

merge_cremad()