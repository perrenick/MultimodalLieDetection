MULTIMODALLIEDETECTION/
├── data/
│   ├── Clips/                <-- Original video clips
│   │   ├── Deceptive/
│   │   └── Truthful/
│   ├── ProcessedFaces/       <-- Cropped face images (by detect_faces.py)
│   │   ├── Deceptive/
│   │   └── Truthful/
│   ├── Audio/                <-- Extracted audio from videos
│   │   ├── Deceptive/
│   │   └── Truthful/
│   ├── AudioSegments/        <-- Optional segmented WAVs (0.5s each)
│   └── ...                   <-- Possibly other intermediate data
│
├── face_recognition/
│   ├── __init__.py
│   ├── detect_faces.py       <-- Face detection & cropping logic (using MTCNN or DSFD, etc.)
│   └── emotion_recognition.py-- Optional code for facial emotion recognition
│
├── audio_recognition/
│   ├── __init__.py
│   └── process_audio.py      <-- Audio extraction, denoising, MFCC+MS, openSMILE, audio emotion SVM
│
├── fusion/
│   ├── __init__.py
│   └── build_EST.py          <-- Voting scheme, EST feature construction (Algorithm 1 & 2 in paper)
│
├── experiments/              <-- (Optional) Jupyter notebooks or scripts for analysis
│   └── demo.ipynb
│
├── main.py                   <-- Orchestrates the entire pipeline (detect faces, audio features, fuse, train/test)
├── test.py                   <-- Unit tests or quick checks
├── requirements.txt          <-- Dependencies (OpenCV, librosa, python_speech_features, sklearn, etc.)
├── README.md                 <-- Detailed usage instructions
└── .gitignore                <-- Ignores venv, data, logs, etc. as needed
