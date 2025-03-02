# import cv2 as cv
# import os 
# from face_recognition.detect_faces import FaceDetector


# # Why extract frames?

# # The paper mentions that raw video must be split into individual frames for processing.
# # Each frame becomes a separate sample for emotion recognition.
# # Helps in tracking gestures, expressions, and deception patterns.

# def preprocess_video(cap, file):
#     # cap = cv.VideoCapture('./Clips/Deceptive/trial_lie_056.mp4')
#     # ref_img_path = './trial_056.jpg'
#     # ref_image = cv.imread(ref_img_path)
#     detector = FaceDetector()

#     frame_id=0

#     while True:
#         ret, frame = cap.read()

#         frame_id+=1
#         if not ret:
#             print('End of video stream')
#             break
        
#         result = detector.people_finder(frame)
        
#         if result==True:
#             print(f'We have more than 1 people in frame {frame_id} in video {file}')
        
#         # processed_frame = detector.detect_faces(frame, frame_id, ref_img_path)

#         # if processed_frame is not None:
#         #     cv.imshow('Frame with faces', processed_frame)
        
#         # if cv.waitKey(1) & 0xFF == ord('q'):
#         #     break
                
#     cap.release()
#     cv.destroyAllWindows()

# if __name__ == '__main__':

#     deceptive_vid = './Clips/Deceptive'
#     truthful_vid = './Clips/Truthful'

#     for file in os.listdir(deceptive_vid):
#         path_to_vid = os.path.join(deceptive_vid, file)
#         cap = cv.VideoCapture(path_to_vid)
#         preprocess_video(cap, file)



# # print(cap.get(cv.CAP_PROP_FRAME_COUNT))
# # print(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
# # print(cap.get(cv.CAP_PROP_FRAME_WIDTH))
# # print(cap.get(cv.CAP_PROP_FPS))

import cv2 as cv
import os
from face_recognition.detect_faces import FaceDetector

def preprocess_video(video_path, file):
    """Checks if more than one person appears in any frame of the video."""
    
    cap = cv.VideoCapture(video_path)
    detector = FaceDetector()
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f'End of video: {file}')
            break

        frame_id += 1
        result = detector.people_finder(frame)
        
        if result:  # Assuming people_finder() returns True when more than one person is detected
            print(f'More than 1 person detected in frame {frame_id} of video {file}')

    cap.release()
    cv.destroyAllWindows()


import os

if __name__ == '__main__':
    deceptive_vid = './Clips/Deceptive'
    truthful_vid = './Clips/Truthful'
    start_file = "trial_lie_005.mp4"  # Define the file to start from
    start_processing = False  # Flag to track when to start processing

    # Process deceptive videos
    for file in sorted(os.listdir(deceptive_vid)):  # Ensure sorted order
        if file == start_file:
            start_processing = True  # Set flag when reaching the target file
        
        if not start_processing:
            continue  # Skip files until reaching 'trial_lie_005.mp4'

        path_to_vid = os.path.join(deceptive_vid, file)
        preprocess_video(path_to_vid, file)  # Call your function


    # Process truthful videos
    # for file in os.listdir(truthful_vid):
    #     if file.endswith(".mp4"):
    #         path_to_vid = os.path.join(truthful_vid, file)
    #         preprocess_video(path_to_vid, file)
