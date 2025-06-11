import cv2
import os
import math
import csv
import mediapipe as mp

# Input/output paths
VIDEO_FOLDER = 'output/annotated_videos'
RESULT_CSV_PATH = 'results/all_angles.csv'
os.makedirs(os.path.dirname(RESULT_CSV_PATH), exist_ok=True)

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Helper to calculate angle
def calculate_angle(a, b, c):
    a = [a.x, a.y]
    b = [b.x, b.y]
    c = [c.x, c.y]

    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]

    cosine = (ba[0]*bc[0] + ba[1]*bc[1]) / (
        math.sqrt(ba[0]**2 + ba[1]**2) * math.sqrt(bc[0]**2 + bc[1]**2)
    )
    angle = math.degrees(math.acos(cosine))
    return angle

# Create CSV and write header
with open(RESULT_CSV_PATH, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Video', 'Frame', 'KneeAngle', 'AnkleAngle'])  # CSV header

    # Process each video
    for filename in os.listdir(VIDEO_FOLDER):
        if filename.endswith('.mp4') or filename.endswith('.avi'):
            video_path = os.path.join(VIDEO_FOLDER, filename)
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"❌ Cannot open: {video_path}")
                continue

            print(f"📹 Processing: {filename}")
            frame_num = 0

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                frame_num += 1
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
                    knee = lm[mp_pose.PoseLandmark.LEFT_KNEE]
                    ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
                    foot = lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]

                    knee_angle = calculate_angle(hip, knee, ankle)
                    ankle_angle = calculate_angle(knee, ankle, foot)

                    writer.writerow([
                        filename,
                        frame_num,
                        round(knee_angle, 2),
                        round(ankle_angle, 2)
                    ])

            cap.release()

print(f"All angle data saved to: {RESULT_CSV_PATH}")
