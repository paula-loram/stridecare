import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt
import pandas as pd
import os
import subprocess
from api.video_angle_processor import build_segment_rotation_matrix, cardanangles, get_landmark_coords

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
    final_columns_order = [
        #'Frame', # Keeping frame for debugging/tracking, will drop before returning final array
        'pelvis_X', 'pelvis_Y', #'pelvis_Z', # Pelvis angles (e.g., Flexion, Abduction, Rotation)
        'L_knee_X', 'L_knee_Y', #'L_knee_Z', # L_knee angles (Flexion, Abduction, Rotation)
        'R_knee_X', 'R_knee_Y',# 'R_knee_Z', # R_knee angles
        'L_hip_X', 'L_hip_Y',# 'L_hip_Z', # L_hip angles
        'R_hip_X', 'R_hip_Y'#, 'R_hip_Z', # R_hip angles
    ]

    angle_data_list = [] # List to store angle dictionaries for each frame
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
            if frame_count % 100 == 0:
                print(f"Processing frame {frame_count}...")
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
            current_frame_angles = {col: np.nan for col in final_columns_order}
            current_frame_angles['Frame'] = frame_count

            if results.pose_world_landmarks: # Use world landmarks for 3D biomechanical angles
                landmarks = results.pose_world_landmarks.landmark

                # Extract MediaPipe Landmarks (using world coordinates)
                r_hip = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
                r_knee = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE)
                r_ankle = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE)
                r_shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)

                l_hip = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
                l_knee = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_KNEE)
                l_ankle = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
                l_shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)

                current_R_segs = {}
                try:
                    # Build Segment Rotation Matrices
                    # Pelvis SCS uses R_HIP, L_HIP, R_SHOULDER, L_SHOULDER
                    current_R_segs['pelvis'] = build_segment_rotation_matrix(
                        r_hip, l_hip, r_shoulder, l_shoulder, segment_type='pelvis'
                    )
                    # Thigh SCS from KNEE (proximal for Z-axis) to HIP (distal for Z-axis)
                    current_R_segs['R_thigh'] = build_segment_rotation_matrix(
                        r_hip, r_knee, segment_type='thigh'
                    )
                    current_R_segs['L_thigh'] = build_segment_rotation_matrix(
                        l_hip, l_knee, segment_type='thigh'
                    )
                    # Shank SCS from ANKLE (proximal for Z-axis) to KNEE (distal for Z-axis)
                    current_R_segs['R_shank'] = build_segment_rotation_matrix(
                        r_knee, r_ankle, segment_type='shank'
                    )
                    current_R_segs['L_shank'] = build_segment_rotation_matrix(
                        l_knee, l_ankle, segment_type='shank'
                    )

                    # --- Calculate Joint Angles ---
                    # A joint angle (e.g., knee) is the relative rotation between two segments (e.g., thigh and shank).
                    # Rotation of distal segment relative to proximal segment (R_distal_wrt_proximal = R_proximal.T @ R_distal)

                    # Pelvis Global Angles (Pelvis SCS relative to MediaPipe World)
                    R_pelvis_global = current_R_segs.get('pelvis')
                    if R_pelvis_global is not None and not np.any(np.isnan(R_pelvis_global)):
                        pelvis_angles_rad = cardanangles(R_pelvis_global)
                        pelvis_angles_deg = np.degrees(pelvis_angles_rad)
                        current_frame_angles['pelvis_X'] = pelvis_angles_deg[0]
                        current_frame_angles['pelvis_Y'] = pelvis_angles_deg[1]
                        current_frame_angles['pelvis_Z'] = pelvis_angles_deg[2] # Include Z-angle

                    # Right Knee: Shank relative to Thigh (R_R_shank_wrt_R_thigh = R_R_thigh.T @ R_R_shank)
                    R_R_thigh = current_R_segs.get('R_thigh')
                    R_R_shank = current_R_segs.get('R_shank')
                    if R_R_thigh is not None and R_R_shank is not None and \
                    not np.any(np.isnan(R_R_thigh)) and not np.any(np.isnan(R_R_shank)):
                        R_R_knee = R_R_thigh.T @ R_R_shank
                        R_knee_angles = np.degrees(cardanangles(R_R_knee))
                        current_frame_angles['R_knee_X'] = R_knee_angles[0]
                        current_frame_angles['R_knee_Y'] = R_knee_angles[1]
                        current_frame_angles['R_knee_Z'] = R_knee_angles[2] # Include Z-angle

                    # Left Knee: Shank relative to Thigh
                    R_L_thigh = current_R_segs.get('L_thigh')
                    R_L_shank = current_R_segs.get('L_shank')
                    if R_L_thigh is not None and R_L_shank is not None and \
                    not np.any(np.isnan(R_L_thigh)) and not np.any(np.isnan(R_L_shank)):
                        R_L_knee = R_L_thigh.T @ R_L_shank
                        L_knee_angles = np.degrees(cardanangles(R_L_knee))
                        current_frame_angles['L_knee_X'] = L_knee_angles[0]
                        current_frame_angles['L_knee_Y'] = L_knee_angles[1]
                        current_frame_angles['L_knee_Z'] = L_knee_angles[2] # Include Z-angle

                    # Right Hip: Thigh relative to Pelvis (R_R_thigh_wrt_pelvis = R_pelvis.T @ R_R_thigh)
                    if R_pelvis_global is not None and R_R_thigh is not None and \
                    not np.any(np.isnan(R_pelvis_global)) and not np.any(np.isnan(R_R_thigh)):
                        R_R_hip = R_pelvis_global.T @ R_R_thigh
                        R_hip_angles = np.degrees(cardanangles(R_R_hip))
                        current_frame_angles['R_hip_X'] = R_hip_angles[0]
                        current_frame_angles['R_hip_Y'] = R_hip_angles[1]
                        current_frame_angles['R_hip_Z'] = R_hip_angles[2] # Include Z-angle

                    # Left Hip: Thigh relative to Pelvis
                    if R_pelvis_global is not None and R_L_thigh is not None and \
                    not np.any(np.isnan(R_pelvis_global)) and not np.any(np.isnan(R_L_thigh)):
                        R_L_hip = R_pelvis_global.T @ R_L_thigh
                        L_hip_angles = np.degrees(cardanangles(R_L_hip))
                        current_frame_angles['L_hip_X'] = L_hip_angles[0]
                        current_frame_angles['L_hip_Y'] = L_hip_angles[1]
                        current_frame_angles['L_hip_Z'] = L_hip_angles[2] # Include Z-angle

                except Exception as e:
                    print(f"Warning: Error calculating angles for frame {frame_count}: {e}")
                    # Values remain NaN as initialized, which is acceptable for missing data

            angle_data_list.append(current_frame_angles) # Append dictionary for this frame

            out.write(image)
            frame_count += 1
        print(f"Total frames written: {frame_count}")
    cap.release()
    out.release()

    if not angle_data_list:
        print("No pose data found for any frame. Returning empty array.")
        return np.array([])

    # Convert collected angle data to a Pandas DataFrame and enforce column order
    final_angles_df = pd.DataFrame(angle_data_list)

    # Ensure all required columns exist, filling with NaN if a column is entirely missing
    for col in final_columns_order:
        if col not in final_angles_df.columns:
            final_angles_df[col] = 999 # Use 999 as a placeholder for missing columns

    # Reindex the DataFrame to ensure the desired column order (important!)
    final_angles_df = final_angles_df[final_columns_order]
    # Remove all column containing name 'Z' as they are not needed
    final_angles_df = final_angles_df.loc[:, ~final_angles_df.columns.str.contains('Z')]
    final_angles_df.drop(columns=['Frame'], inplace=True, errors='ignore')  # Drop 'Frame' column if it exists
    final_angles_df = final_angles_df[final_columns_order]  # Ensure correct order

    print(f'Final angles DataFrame shape: {final_angles_df.shape}')
    print(f'Final angles DataFrame columns: {final_angles_df.columns.tolist()}')
    # Save the DataFrame to a CSV file for debugging
    csv_output_path = output_path.replace('.mp4', '_angles.csv')
    final_angles_df.to_csv(csv_output_path, index=False)
    print(f"Output video size: {os.path.getsize(output_path)} bytes")
    h264_output_path = output_path.replace('.mp4', '_h264.mp4')
    success = convert_to_h264(output_path, h264_output_path)
    if success:
        # remove the original output file if conversion was successful
        os.remove(output_path)
        print(f"H.264 video saved to {h264_output_path}")
        return h264_output_path, final_angles_df.values
    else:
        print("Failed to convert video to H.264.")
        return output_path, final_angles_df.values  # fallback

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
