import cv2
import mediapipe as mp
import os

# Set folder with videos
VIDEO_FOLDER = 'raw_data/running'
OUTPUT_FOLDER = 'output/annotated_videos'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# Loop through all video files
    for filename in os.listdir(VIDEO_FOLDER):
        if filename.endswith('.avi') or filename.endswith('.mp4'):
            video_path = os.path.join(VIDEO_FOLDER, filename)
            print(f"Processing: {video_path}")

            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            output_path = os.path.join(OUTPUT_FOLDER, f"annotated_{filename}")
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                annotated_image = frame.copy()

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS
                    )

                out.write(annotated_image)

            cap.release()
            out.release()
            print(f"Saved: {output_path}")

cv2.destroyAllWindows()
