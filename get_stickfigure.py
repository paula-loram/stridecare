import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt
import pandas as pd

#function to get mediapipe pose stickfigures out of a video file
def get_stickfigure(video_path: str):
# Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    #get video
    video_path = video_path 
    cap = cv2.VideoCapture(video_path)

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2, # Use model_complexity=2 for best accuracy, but slower
        enable_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks and results.pose_world_landmarks:
                landmarks = results.pose_world_landmarks.landmark

                # Draw the pose annotation on the image.
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            cv2.imshow('MediaPipe Pose - Cardan Angles', image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
