import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt
import pandas as pd
import os
import subprocess

#function to get mediapipe pose stickfigures out of a video file
def get_stickfigure(video_path: str, output_path: str = "stickfigure_output.mp4") -> str:
    import cv2
    import mediapipe as mp

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"DEBUG: fps={fps}, width={width}, height={height}")
    if fps == 0 or width == 0 or height == 0:
        print("ERROR: Invalid video properties. Cannot save output video.")
        cap.release()
        return None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for better compatibility
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print("ERROR: VideoWriter failed to open. Check codec and output path.")
        cap.release()
        return None

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            out.write(image)
            frame_count += 1
        print(f"Total frames written: {frame_count}")
    cap.release()
    out.release()
    print(f"Output video size: {os.path.getsize(output_path)} bytes")
    h264_output_path = output_path.replace('.mp4', '_h264.mp4')
    success = convert_to_h264(output_path, h264_output_path)
    if success:
        print(f"H.264 video saved to {h264_output_path}")
        return h264_output_path
    else:
        print("Failed to convert video to H.264.")
        return output_path  # fallback

def convert_to_h264(input_path: str, output_path: str):
    """
    Converts a video file to H.264 codec using ffmpeg.
    TO ensure ffmpeg is installed, run:
    `pip install imageio[ffmpeg]` or install ffmpeg manually.
    Alternatively,  sudo apt update
                    sudo apt install ffmpeg libavcodec-extra on linux systems.
    """
    command = [
        "ffmpeg",
        "-y",  # Overwrite output file if it exists
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("FFmpeg error:", result.stderr.decode())
        return False
    return True
