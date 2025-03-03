#!/usr/bin/env python3

import os
import cv2 as cv
import numpy as np
from tqdm import tqdm

# Face detection & emotion recognition
from face_recognition.detect_faces import FaceDetector
from face_recognition.emotion_recognition import FaceEmotionClassifier

# Audio processing & emotion recognition
from audio_recognition.process_audio import (
    extract_audio_from_video,
    denoise_audio,
    segment_audio,
    compute_mfcc_plus_ms,
    # possibly run_opensmile_is13, parse_opensmile_csv, etc.
)
from audio_recognition.process_audio import AudioEmotionClassifier

# Fusion & EST
from fusion.build_EST import fuse_audio_visual_emotions, build_est_feature

########################################################################
# CONFIG
########################################################################

INPUT_VIDEO_DIR = "./data/Clips"
OUTPUT_FACES_DIR = "./data/ProcessedFaces"
AUDIO_DIR = "./data/Audio"
AUDIO_SEGMENTS_DIR = "./data/AudioSegments"
FINAL_FEATURES_CSV = "./EST_features.csv"

CATEGORIES = ["Deceptive", "Truthful"]  # subfolders

########################################################################
# MAIN PIPELINE
########################################################################

def main():
    # 1) FACE DETECTION / CROPPING
    # ---------------------------------------------------------------------
    print("=== STEP 1: Face Detection & Cropping ===")
    os.makedirs(OUTPUT_FACES_DIR, exist_ok=True)

    face_detector = FaceDetector(min_confidence=0.99, min_size=30)

    for category in CATEGORIES:
        input_cat_dir = os.path.join(INPUT_VIDEO_DIR, category)
        output_cat_dir = os.path.join(OUTPUT_FACES_DIR, category)
        os.makedirs(output_cat_dir, exist_ok=True)

        for video_file in os.listdir(input_cat_dir):
            if not video_file.endswith(".mp4"):
                continue

            video_path = os.path.join(input_cat_dir, video_file)
            video_name = os.path.splitext(video_file)[0]
            video_output_dir = os.path.join(output_cat_dir, video_name)
            os.makedirs(video_output_dir, exist_ok=True)

            print(f"\n[INFO] Processing (faces): {video_file}")

            cap = cv.VideoCapture(video_path)
            total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            frame_id = 0

            with tqdm(total=total_frames, desc=f"⏳ {video_file}", unit="frame") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    face_crop = face_detector.detect_largest_face(frame)
                    if face_crop is not None and face_crop.size > 0:
                        # Save cropped face
                        frame_filename = os.path.join(
                            video_output_dir, f"frame_{frame_id:04d}.jpg"
                        )
                        cv.imwrite(frame_filename, face_crop)

                    frame_id += 1
                    pbar.update(1)
            cap.release()

    print("[INFO] Face Detection complete. Cropped faces saved in:", OUTPUT_FACES_DIR)

    # 2) AUDIO EXTRACTION & DENOISING (OPTIONAL)
    # ---------------------------------------------------------------------
    print("\n=== STEP 2: Extract & (Optionally) Denoise Audio ===")
    os.makedirs(AUDIO_DIR, exist_ok=True)

    for category in CATEGORIES:
        vid_cat_dir = os.path.join(INPUT_VIDEO_DIR, category)
        aud_cat_dir = os.path.join(AUDIO_DIR, category)
        os.makedirs(aud_cat_dir, exist_ok=True)

        for video_file in os.listdir(vid_cat_dir):
            if not video_file.endswith(".mp4"):
                continue

            video_path = os.path.join(vid_cat_dir, video_file)
            audio_name = os.path.splitext(video_file)[0] + ".wav"
            audio_path = os.path.join(aud_cat_dir, audio_name)

            if not os.path.exists(audio_path):
                extract_audio_from_video(video_path, audio_path)

            # If you want to denoise using SoX:
            # noise_profile = "./some_noise.prof"
            # denoised_path = audio_path.replace(".wav", "_dn.wav")
            # denoise_audio(audio_path, denoised_path, noise_profile, 0.21)

    print("[INFO] Audio extracted into:", AUDIO_DIR)

    # 3) TRAIN OR LOAD AUDIO EMOTION MODEL
    # ---------------------------------------------------------------------
    print("\n=== STEP 3: Train or Load Audio Emotion Classifier ===")
    audio_emotion_model = AudioEmotionClassifier()
    # EITHER train from scratch using a labeled emotion dataset
    # or load a pre-trained model from disk:
    # audio_emotion_model.fit(X_train, y_train)
    # or:
    # audio_emotion_model = load_pretrained_audio_model("audio_emotion_svm.pkl")

    # 4) TRAIN OR LOAD FACE EMOTION MODEL (OPTIONAL)
    # ---------------------------------------------------------------------
    print("\n=== STEP 4: Train or Load Face Emotion Classifier ===")
    face_emotion_model = FaceEmotionClassifier()
    # Similar approach:
    # face_emotion_model.fit(X_face_train, y_face_train)
    # or:
    # face_emotion_model = load_pretrained_face_model("face_emotion_cnn.pkl")

    # 5) EMOTION INFERENCE + FUSION => EST FEATURE
    # ---------------------------------------------------------------------
    print("\n=== STEP 5: Build EST Features ===")
    # We'll keep track of the final features and labels (Deceptive/Truthful)
    all_features = []
    all_labels = []

    for category in CATEGORIES:
        # 1 if Deceptive, 0 if Truthful
        label = 1 if category == "Deceptive" else 0

        # directories
        vid_cat_dir = os.path.join(INPUT_VIDEO_DIR, category)
        face_cat_dir = os.path.join(OUTPUT_FACES_DIR, category)
        aud_cat_dir = os.path.join(AUDIO_DIR, category)

        for video_file in os.listdir(vid_cat_dir):
            if not video_file.endswith(".mp4"):
                continue

            video_name = os.path.splitext(video_file)[0]
            video_path = os.path.join(vid_cat_dir, video_file)
            face_video_dir = os.path.join(face_cat_dir, video_name)
            audio_path = os.path.join(aud_cat_dir, video_name + ".wav")

            # a) Get face-based emotions for each frame
            frame_files = sorted(
                f for f in os.listdir(face_video_dir) if f.endswith(".jpg")
            )
            visual_emotions = []
            for frame_file in frame_files:
                frame_path = os.path.join(face_video_dir, frame_file)
                # predict face emotion
                face_emotion = face_emotion_model.predict_emotion(frame_path)
                visual_emotions.append(face_emotion)

            # b) Segment audio & predict audio emotions
            segments, sr = segment_audio(audio_path, segment_len=0.5)
            audio_emotions = []
            for seg in segments:
                feat_vec = compute_mfcc_plus_ms(seg, sr)  # or openSMILE
                pred_emotion = audio_emotion_model.predict_single(feat_vec)
                audio_emotions.append(pred_emotion)

            # c) Expand audio emotions to match frames (paper: 15 frames per 0.5s if 30 FPS)
            audio_emotions_expanded = []
            frames_per_seg = 15
            for ae in audio_emotions:
                audio_emotions_expanded.extend([ae] * frames_per_seg)
            # If there's mismatch in length, you can trim/pad
            # for simplicity, let's do:
            len_diff = len(visual_emotions) - len(audio_emotions_expanded)
            if len_diff > 0:
                # pad audio with last emotion
                audio_emotions_expanded += [audio_emotions_expanded[-1]] * len_diff
            elif len_diff < 0:
                # trim audio array
                audio_emotions_expanded = audio_emotions_expanded[: len(visual_emotions)]

            # d) Fuse to get final "revised" emotion per frame
            revised_emotions = fuse_audio_visual_emotions(
                visual_emotions, audio_emotions_expanded
            )

            # e) Build EST feature from revised_emotions
            est_feat = build_est_feature(revised_emotions)
            # Now you have a 49-dim vector describing the transitions

            # f) Possibly combine with openSMILE or micro-expressions (MEg)
            # me_features = ...
            # is13_features = ...
            # combined_feat = np.concatenate([est_feat, me_features, is13_features])
            combined_feat = est_feat  # placeholder

            all_features.append(combined_feat)
            all_labels.append(label)

    all_features = np.array(all_features)
    all_labels = np.array(all_labels)

    # 6) TRAIN/TEST DECEPTION CLASSIFIER
    # ---------------------------------------------------------------------
    print("\n=== STEP 6: Train & Evaluate Deception Classifier ===")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)

    scores = cross_val_score(clf, all_features, all_labels, cv=10, scoring="roc_auc")
    print(f"[RESULT] 10-fold CV: ROC-AUC = {scores.mean():.4f} (±{scores.std():.4f})")

    # If you want to do a final train on the entire dataset:
    clf.fit(all_features, all_labels)
    # You could then save the model with joblib, etc.

    print("\n[INFO] Done! Final features shape:", all_features.shape)
    print("[INFO] Example AUC =>", scores.mean())


if __name__ == "__main__":
    main()
