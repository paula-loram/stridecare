import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # For storing data

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def calculate_angle(a, b, c):
    """Calculates the 3D angle between three points (a, b, c) where b is the vertex."""
    a = np.array(a) # First point (e.g., hip for knee angle)
    b = np.array(b) # Mid point / Vertex (e.g., knee for knee angle)
    c = np.array(c) # End point (e.g., ankle for knee angle)

    # Create vectors
    ba = a - b
    bc = c - b

    # Calculate cosine of the angle using dot product
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    # Ensure cosine_angle is within [-1, 1] to avoid arccos errors due to floating point inaccuracies
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle_radians = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees

# --- Main processing ---
video_path = 'raw_data/runner_vide.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    exit()

# Store angles for plotting
frame_numbers = []
right_hip_angles = []
left_hip_angles = []
right_knee_angles = []
left_knee_angles = []
right_ankle_angles = []
left_ankle_angles = []
right_foot_angles = []
left_foot_angles = []

# Use MediaPipe Pose
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
        # Convert the BGR image to RGB.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False # Make image not writeable for performance

        # Process the image and find poses
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Convert back to BGR for OpenCV display

        if results.pose_landmarks:
            landmarks = results.pose_world_landmarks.landmark # Use world landmarks for 3D angles

            # Get landmark coordinates for right leg
            r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
            r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
            r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]
            r_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y,
                      landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].z]
            r_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y,
                            landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].z]
            r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]

            # Get landmark coordinates for left leg
            l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                     landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
            l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
            l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                       landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]
            l_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y,
                      landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].z]
            l_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y,
                            landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].z]
            l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]

            # Calculate angles
            try:
                # Hip angle: Using shoulder, hip, knee to define.
                # This measures flexion/extension. A straight upright body would be around 180 degrees.
                # Adjust if you need a different interpretation (e.g., angle with vertical).
                right_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)
                left_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)

                # Knee angle: Using hip, knee, ankle.
                # A straight leg is ~180 degrees. A bent knee is smaller.
                right_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
                left_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)

                # Ankle angle: Using knee, ankle, heel.
                # This represents dorsiflexion/plantarflexion.
                right_ankle_angle = calculate_angle(r_knee, r_ankle, r_heel)
                left_ankle_angle = calculate_angle(l_knee, l_ankle, l_heel)

                # Foot angle: Using ankle, heel, foot_index.
                # This angle captures the orientation of the foot itself.
                right_foot_angle = calculate_angle(r_ankle, r_heel, r_foot_index)
                left_foot_angle = calculate_angle(l_ankle, l_heel, l_foot_index)


                # Store data
                frame_numbers.append(frame_count)
                right_hip_angles.append(right_hip_angle)
                left_hip_angles.append(left_hip_angle)
                right_knee_angles.append(right_knee_angle)
                left_knee_angles.append(left_knee_angle)
                right_ankle_angles.append(right_ankle_angle)
                left_ankle_angles.append(left_ankle_angle)
                right_foot_angles.append(right_foot_angle)
                left_foot_angles.append(left_foot_angle)

                # Optional: Display angles on the video feed (2D projection)
                # Convert world coordinates back to image coordinates for drawing
                r_hip_2d = mp_drawing._normalized_to_pixel_coordinates(
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                    frame.shape[1], frame.shape[0])
                r_knee_2d = mp_drawing._normalized_to_pixel_coordinates(
                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                    frame.shape[1], frame.shape[0])
                r_ankle_2d = mp_drawing._normalized_to_pixel_coordinates(
                    landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
                    frame.shape[1], frame.shape[0])

                if r_knee_2d:
                    cv2.putText(image, f"R Knee: {int(right_knee_angle)}",
                                (r_knee_2d[0], r_knee_2d[1] - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                if r_hip_2d:
                    cv2.putText(image, f"R Hip: {int(right_hip_angle)}",
                                (r_hip_2d[0], r_hip_2d[1] - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                if r_ankle_2d:
                    cv2.putText(image, f"R Ankle: {int(right_ankle_angle)}",
                                (r_ankle_2d[0], r_ankle_2d[1] - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)


            except Exception as e:
                # This can happen if a landmark is not detected
                # print(f"Error calculating angle: {e} at frame {frame_count}")
                pass # Continue processing

            # Draw the pose annotation on the image.
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        cv2.imshow('MediaPipe Pose - Angle Calculation', image)

        if cv2.waitKey(5) & 0xFF == 27: # Press 'Esc' to exit
            break

cap.release()
cv2.destroyAllWindows()

# --- Plotting the angles ---
if frame_numbers:
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 2, 1)
    plt.plot(frame_numbers, right_hip_angles, label='Right Hip Angle')
    plt.xlabel('Frame')
    plt.ylabel('Angle (degrees)')
    plt.title('Right Hip Angle Over Time')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(frame_numbers, left_hip_angles, label='Left Hip Angle')
    plt.xlabel('Frame')
    plt.ylabel('Angle (degrees)')
    plt.title('Left Hip Angle Over Time')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(frame_numbers, right_knee_angles, label='Right Knee Angle')
    plt.xlabel('Frame')
    plt.ylabel('Angle (degrees)')
    plt.title('Right Knee Angle Over Time')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(frame_numbers, left_knee_angles, label='Left Knee Angle')
    plt.xlabel('Frame')
    plt.ylabel('Angle (degrees)')
    plt.title('Left Knee Angle Over Time')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.plot(frame_numbers, right_ankle_angles, label='Right Ankle Angle')
    plt.xlabel('Frame')
    plt.ylabel('Angle (degrees)')
    plt.title('Right Ankle Angle Over Time')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.plot(frame_numbers, left_ankle_angles, label='Left Ankle Angle')
    plt.xlabel('Frame')
    plt.ylabel('Angle (degrees)')
    plt.title('Left Ankle Angle Over Time')
    plt.grid(True)
    plt.legend()

    # You can add similar subplots for foot angles if desired.
    # plt.subplot(3, 2, 5)
    # plt.plot(frame_numbers, right_foot_angles, label='Right Foot Angle')
    # plt.xlabel('Frame')
    # plt.ylabel('Angle (degrees)')
    # plt.title('Right Foot Angle Over Time')
    # plt.grid(True)
    # plt.legend()

    # plt.subplot(3, 2, 6)
    # plt.plot(frame_numbers, left_foot_angles, label='Left Foot Angle')
    # plt.xlabel('Frame')
    # plt.ylabel('Angle (degrees)')
    # plt.title('Left Foot Angle Over Time')
    # plt.grid(True)
    # plt.legend()

    plt.tight_layout()
    plt.show()

    # Optional: Save data to a CSV file
    df = pd.DataFrame({
        'Frame': frame_numbers,
        'Right_Hip_Angle': right_hip_angles,
        'Left_Hip_Angle': left_hip_angles,
        'Right_Knee_Angle': right_knee_angles,
        'Left_Knee_Angle': left_knee_angles,
        'Right_Ankle_Angle': right_ankle_angles,
        'Left_Ankle_Angle': left_ankle_angles,
        'Right_Foot_Angle': right_foot_angles,
        'Left_Foot_Angle': left_foot_angles
    })
    df.to_csv('joint_angles.csv', index=False)
    print("Joint angles saved to joint_angles.csv")
else:
    print("No pose data found to plot.")
