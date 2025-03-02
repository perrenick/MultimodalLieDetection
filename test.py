import cv2 as cv
from face_recognition.detect_faces import FaceDetector

video_path = "./Clips/Deceptive/trial_lie_026.mp4"
detector = FaceDetector()

cap = cv.VideoCapture(video_path)
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print(f'End of video')
        break

    largest_face, landmark = detector.detect_largest_face(frame)

    if largest_face is not None:
        x1, y1, x2, y2 = largest_face
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box for target face

    cv.imshow("Face Detection", frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
