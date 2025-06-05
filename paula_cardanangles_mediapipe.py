import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.transform import Rotation as R_scipy # Renamed to avoid conflict

# --- YOUR PROVIDED CARDAN ANGLES FUNCTION ---
def cardanangles(R3):
    """
    Compute Cardan (XYZ) angles from a 3×3 rotation matrix R3.
    Returns [Rx, Ry, Rz] in radians, following the same conventions
    as your MATLAB `cardanangles`:

      | Cz*Cy - Sz*Sy*Sx   Sz*Cy + Cz*Sy*Sx  -Sy*Cx |
      | -Sz*Cx             Cz*Cx              Sx     |
      | Cz*Sy + Sz*Cy*Sx   Sz*Sy - Cz*Cy*Sx  Cy*Cx  |

    That is: Rx about X, Ry about Y, Rz about Z.
    """
    # Extract relevant sine/cosine terms
    Sx = R3[2, 1]
    # Clamp for numerical stability
    Sx = np.clip(Sx, -1.0, 1.0)
    Cx = np.sqrt(1 - Sx**2)

    # Gimbal‐lock check
    if np.isclose(Cx, 0.0):
        # When Cx ≈ 0, we lose ability to uniquely find Rx vs Rz
        # Fallback: set Rz = 0, solve Rx from off‐diagonals
        Rx = np.arctan2(-R3[1, 2], R3[1, 1])
        Ry = np.arcsin(Sx)
        Rz = 0.0
    else:
        # These formulas extract Rx, Ry, Rz from the given matrix structure.
        # They correspond to a specific order (e.g., ZYX or XYZ based on how matrix was formed)
        Rx = np.arctan2(R3[2, 1], R3[1, 1]) # Based on the matrix, this should be atan2(Sx, Cz*Cx) / atan2(Sx, Cx)
                                            # Your original was atan2(Sx, R3[1,1]) which is correct if R3[1,1] is Cz*Cx
        Ry = np.arctan2(-R3[0, 2], R3[2, 2]) # Corresponds to atan2(-Sy*Cx, Cy*Cx) = atan2(-Sy, Cy)
        Rz = np.arctan2(-R3[1, 0], R3[1, 1]) # Corresponds to atan2(-Sz*Cx, Cz*Cx) = atan2(-Sz, Cz)

    return np.array([Rx, Ry, Rz])

# --- Helper function for MediaPipe Landmark Coordinates ---
def get_landmark_coords(landmarks, landmark_enum):
    """Helper to get 3D coords from landmark object."""
    lm = landmarks[landmark_enum.value]
    # MediaPipe world landmarks are in meters, origin at hip
    return np.array([lm.x, lm.y, lm.z])

# --- Function to build a segment's local rotation matrix ---
def build_segment_rotation_matrix(proximal_lm, distal_lm, ref_lm_for_x=None, ref_lm_for_y=None, segment_type=''):
    """
    Builds a 3x3 rotation matrix for a segment's local coordinate system (SCS)
    relative to the MediaPipe world coordinate system.

    Args:
        proximal_lm (np.array): 3D coordinates of the proximal landmark.
        distal_lm (np.array): 3D coordinates of the distal landmark.
        ref_lm_for_x (np.array, optional): A third landmark to help define the X-axis (e.g., opposite hip).
        ref_lm_for_y (np.array, optional): A third landmark to help define the Y-axis.
        segment_type (str): 'pelvis', 'thigh', 'shank', 'foot' for specific axis definitions.

    Returns:
        np.array: A 3x3 rotation matrix (columns are X, Y, Z axes of the segment).
    """
    # Initialize axes as zeros
    X_axis = np.zeros(3)
    Y_axis = np.zeros(3)
    Z_axis = np.zeros(3)

    if segment_type == 'pelvis':
        # Pelvis Z-axis (vertical, roughly up from mid-hip)
        Z_axis = (proximal_lm + distal_lm) / 2 # Midpoint of hips for origin if needed, but for axis it's usually based on spine/hips
        # A simple Z for pelvis could be perpendicular to the plane formed by both hips and one shoulder.
        # Or, vertical in the MediaPipe frame (which is Z pointing away from camera, Y is vertical)
        # Let's define the pelvis axes roughly similar to a common biomechanical model:
        # X: Anterior-posterior (forward)
        # Y: Medio-lateral (side-to-side)
        # Z: Longitudinal (vertical)

        # Assuming 'proximal_lm' is R_HIP and 'distal_lm' is L_HIP
        # Z-axis: Vector from mid-hip to mid-shoulder or just vertical if we assume upright
        # Let's use the average of shoulders and hips to get a robust vertical.
        # This function takes only two points and optional references. Let's adapt.
        # For pelvis, use R_HIP, L_HIP, R_SHOULDER, L_SHOULDER
        if ref_lm_for_x is not None and ref_lm_for_y is not None: # Here, ref_lm_for_x = R_shoulder, ref_lm_for_y = L_shoulder
            mid_hip = (proximal_lm + distal_lm) / 2 # R_HIP + L_HIP / 2
            mid_shoulder = (ref_lm_for_x + ref_lm_for_y) / 2 # R_SHOULDER + L_SHOULDER / 2

            # Z-axis (longitudinal, roughly superior)
            Z_axis = mid_shoulder - mid_hip
            Z_axis = Z_axis / np.linalg.norm(Z_axis) if np.linalg.norm(Z_axis) != 0 else np.array([0.,1.,0.])

            # Y-axis (medio-lateral, roughly right to left from subject's view)
            # Use L_HIP - R_HIP to get a vector pointing left.
            Y_axis = distal_lm - proximal_lm # L_HIP - R_HIP
            Y_axis = Y_axis / np.linalg.norm(Y_axis) if np.linalg.norm(Y_axis) != 0 else np.array([0.,0.,1.])

            # X-axis (anterior-posterior, cross product of Y and Z)
            X_axis = np.cross(Y_axis, Z_axis)
            X_axis = X_axis / np.linalg.norm(X_axis) if np.linalg.norm(X_axis) != 0 else np.array([1.,0.,0.])

            # Re-orthogonalize Y_axis
            Y_axis = np.cross(Z_axis, X_axis)
            Y_axis = Y_axis / np.linalg.norm(Y_axis) if np.linalg.norm(Y_axis) != 0 else np.array([0.,1.,0.])

        else: # Fallback if not all pelvis points provided
            # MediaPipe's default Z is depth, Y is vertical. Let's make pelvis Z be vertical.
            # X_axis = np.array([1., 0., 0.]) # Assuming MediaPipe X is roughly medio-lateral
            # Y_axis = np.array([0., 1., 0.]) # MediaPipe Y is roughly superior (up)
            # Z_axis = np.array([0., 0., 1.]) # MediaPipe Z is depth (anterior-posterior)
            # This would define a "global" pelvis, not subject-relative.
            # For pelvis, we really need more points or a fixed assumption relative to MediaPipe's global frame.
            # Let's define it such that X is medio-lateral, Y is anterior-posterior, Z is longitudinal (vertical).
            # This is hard without more markers or a consistent anatomical plane definition.
            # For now, let's make it align with MediaPipe's axis and adjust interpretation if needed.
            # MediaPipe's global frame is: +X to subject's right, +Y downwards, +Z into the camera.
            # Let's try to align pelvis axes to a common biomechanical setup:
            # X: Flexion-Extension (medial-lateral, pointing right)
            # Y: Abduction-Adduction (anterior-posterior, pointing forward)
            # Z: Internal-External Rotation (longitudinal, pointing up)

            # Pelvis Z (longitudinal/vertical): from mid-hip up (MediaPipe's Y is down)
            # So, vertical is -MediaPipe Y.
            Z_axis = np.array([0., -1., 0.]) # Aligned with global up/down

            # Pelvis X (medio-lateral): from left hip to right hip (MediaPipe's X is right)
            X_axis = np.array([1., 0., 0.]) # Aligned with global right/left

            # Pelvis Y (anterior-posterior): cross product of Z and X (front/back)
            Y_axis = np.cross(Z_axis, X_axis) # Should be towards camera (positive Z in MediaPipe)
            Y_axis = Y_axis / np.linalg.norm(Y_axis) if np.linalg.norm(Y_axis) != 0 else np.array([0.,0.,1.])

            # Recalculate X to ensure orthogonality
            X_axis = np.cross(Y_axis, Z_axis)
            X_axis = X_axis / np.linalg.norm(X_axis) if np.linalg.norm(X_axis) != 0 else np.array([1.,0.,0.])

    elif segment_type in ['thigh', 'shank']:
        # Z-axis (longitudinal): Vector from distal to proximal landmark
        # E.g., for thigh: from knee to hip. For shank: from ankle to knee.
        Z_axis = (proximal_lm - distal_lm)
        Z_axis = Z_axis / np.linalg.norm(Z_axis) if np.linalg.norm(Z_axis) != 0 else np.array([0.,0.,1.])

        # X-axis (medio-lateral): Approximation using cross product with a global direction.
        # This is the trickiest part for single-camera data.
        # We assume X-axis is roughly perpendicular to Z and in the horizontal plane (or coronal plane)
        # Global 'up' in MediaPipe is -Y. Global 'right' is +X. Global 'forward' is -Z.
        # Let's try to define X for knee flexion (roughly medio-lateral) by crossing longitudinal with 'forward' or 'right'
        # For the knee, the flexion axis is roughly medio-lateral.
        # For thigh/shank, let's try to define X as roughly horizontal and perpendicular to the bone.
        # If the segment is mostly vertical, cross with global X (right). If mostly horizontal, cross with global Y (up).

        # A common way to get medio-lateral for thigh/shank is to use medial/lateral epicondyles.
        # Since we don't have them, we use cross product with global Y-axis to define a roughly medio-lateral axis.
        # MediaPipe Y is down. So global UP is [0, -1, 0].
        global_up = np.array([0., -1., 0.])
        X_axis = np.cross(Z_axis, global_up) # This gives a vector perpendicular to longitudinal and vertical
        X_axis = X_axis / np.linalg.norm(X_axis) if np.linalg.norm(X_axis) != 0 else np.array([1.,0.,0.])

        # Y-axis (anterior-posterior): cross product of Z and X
        Y_axis = np.cross(Z_axis, X_axis)
        Y_axis = Y_axis / np.linalg.norm(Y_axis) if np.linalg.norm(Y_axis) != 0 else np.array([0.,0.,1.])

        # Re-orthogonalize X_axis
        X_axis = np.cross(Y_axis, Z_axis)
        X_axis = X_axis / np.linalg.norm(X_axis) if np.linalg.norm(X_axis) != 0 else np.array([1.,0.,0.])


    elif segment_type == 'foot':
        # Z-axis (longitudinal): From heel to foot_index (along the foot length)
        Z_axis = (ref_lm_for_x - distal_lm) # Foot_index - Ankle/Heel
        Z_axis = Z_axis / np.linalg.norm(Z_axis) if np.linalg.norm(Z_axis) != 0 else np.array([0.,0.,1.])

        # X-axis (medio-lateral/vertical for ankle rotation): Cross product with longitudinal
        # Ankle complex is tricky. Often X is the flexion/extension axis (medio-lateral, like knee)
        # Y is inversion/eversion, Z is abduction/adduction.
        # Let's try to define X as medio-lateral using a vertical reference
        global_up = np.array([0., -1., 0.])
        X_axis = np.cross(Z_axis, global_up)
        X_axis = X_axis / np.linalg.norm(X_axis) if np.linalg.norm(X_axis) != 0 else np.array([1.,0.,0.])

        Y_axis = np.cross(Z_axis, X_axis)
        Y_axis = Y_axis / np.linalg.norm(Y_axis) if np.linalg.norm(Y_axis) != 0 else np.array([0.,1.,0.])

        X_axis = np.cross(Y_axis, Z_axis)
        X_axis = X_axis / np.linalg.norm(X_axis) if np.linalg.norm(X_axis) != 0 else np.array([1.,0.,0.])

    else:
        raise ValueError(f"Unknown segment type: {segment_type}")

    # Return rotation matrix where columns are X, Y, Z axes of the segment
    return np.column_stack((X_axis, Y_axis, Z_axis))


# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Main processing ---
video_path = 'raw_data/runner_vid.mp4'  # <<< REPLACE WITH YOUR VIDEO FILE PATH
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    exit()

# List to store rotation matrices for each segment
R_segs_per_frame = {}
segments = ['L_foot', 'R_foot', 'L_shank', 'R_shank', 'L_thigh', 'R_thigh', 'pelvis']
joints = ['L_ankle', 'R_ankle', 'L_knee', 'R_knee', 'L_hip', 'R_hip']

# Initialize dictionaries to store angles and rotation matrices for each frame
angle_data = {joint: [] for joint in joints}
R_joint_data = {joint: [] for joint in joints} # To store the relative joint rotation matrices

# Store raw segment rotation matrices (R_segs) for plotting if needed
R_segment_data = {seg: [] for seg in segments}

# Store frame numbers
frame_numbers = []

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

            # Extract MediaPipe Landmarks
            # Right Leg
            r_hip = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
            r_knee = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE)
            r_ankle = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE)
            r_heel = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_HEEL)
            r_foot_index = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)
            r_shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)

            # Left Leg
            l_hip = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
            l_knee = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_KNEE)
            l_ankle = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
            l_heel = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_HEEL)
            l_foot_index = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_FOOT_INDEX)
            l_shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)

            # --- STEP A/B/C/D/E/F: Build each segment's 3x3 rotation matrix (R_segs) ---
            # This replaces the complex Söderkvist steps with direct SCS definition.

            current_R_segs = {}
            try:
                # Pelvis Rotation Matrix
                # Using R_HIP, L_HIP for proximal/distal. R_SHOULDER, L_SHOULDER for references.
                current_R_segs['pelvis'] = build_segment_rotation_matrix(
                    r_hip, l_hip, r_shoulder, l_shoulder, segment_type='pelvis'
                )

                # Right Thigh Rotation Matrix
                current_R_segs['R_thigh'] = build_segment_rotation_matrix(
                    r_hip, r_knee, segment_type='thigh'
                )
                # Left Thigh Rotation Matrix
                current_R_segs['L_thigh'] = build_segment_rotation_matrix(
                    l_hip, l_knee, segment_type='thigh'
                )

                # Right Shank Rotation Matrix
                current_R_segs['R_shank'] = build_segment_rotation_matrix(
                    r_knee, r_ankle, segment_type='shank'
                )
                # Left Shank Rotation Matrix
                current_R_segs['L_shank'] = build_segment_rotation_matrix(
                    l_knee, l_ankle, segment_type='shank'
                )

                # Right Foot Rotation Matrix
                current_R_segs['R_foot'] = build_segment_rotation_matrix(
                    r_ankle, r_heel, r_foot_index, segment_type='foot' # ankle, heel as main points, foot_index as reference
                )
                # Left Foot Rotation Matrix
                current_R_segs['L_foot'] = build_segment_rotation_matrix(
                    l_ankle, l_heel, l_foot_index, segment_type='foot'
                )

                # Store segment rotation matrices for this frame
                for seg_name in segments:
                    R_segment_data[seg_name].append(current_R_segs[seg_name])

                # --- STEP J: Compute inter‐segment (joint) rotation matrices, then joint angles ---
                # R_shank^T * R_foot  → ankle;  R_thigh^T * R_shank → knee;  R_pelvis^T * R_thigh → hip

                # L_ankle: Shank (proximal) to Foot (distal)
                R_L_ankle = current_R_segs['L_shank'].T @ current_R_segs['L_foot']
                R_joint_data['L_ankle'].append(R_L_ankle)
                angle_data['L_ankle'].append(np.degrees(cardanangles(R_L_ankle)))

                # R_ankle: Shank (proximal) to Foot (distal)
                R_R_ankle = current_R_segs['R_shank'].T @ current_R_segs['R_foot']
                R_joint_data['R_ankle'].append(R_R_ankle)
                angle_data['R_ankle'].append(np.degrees(cardanangles(R_R_ankle)))

                # L_knee: Thigh (proximal) to Shank (distal)
                R_L_knee = current_R_segs['L_thigh'].T @ current_R_segs['L_shank']
                R_joint_data['L_knee'].append(R_L_knee)
                angle_data['L_knee'].append(np.degrees(cardanangles(R_L_knee)))

                # R_knee: Thigh (proximal) to Shank (distal)
                R_R_knee = current_R_segs['R_thigh'].T @ current_R_segs['R_shank']
                R_joint_data['R_knee'].append(R_R_knee)
                angle_data['R_knee'].append(np.degrees(cardanangles(R_R_knee)))

                # L_hip: Pelvis (proximal) to Thigh (distal)
                R_L_hip = current_R_segs['pelvis'].T @ current_R_segs['L_thigh']
                R_joint_data['L_hip'].append(R_L_hip)
                angle_data['L_hip'].append(np.degrees(cardanangles(R_L_hip)))

                # R_hip: Pelvis (proximal) to Thigh (distal)
                R_R_hip = current_R_segs['pelvis'].T @ current_R_segs['R_thigh']
                R_joint_data['R_hip'].append(R_R_hip)
                angle_data['R_hip'].append(np.degrees(cardanangles(R_R_hip)))

                frame_numbers.append(frame_count)

            except Exception as e:
                # Handle cases where landmarks might be missing or definition leads to singular matrix
                # print(f"Could not compute segment or joint rotation for frame {frame_count}: {e}")
                for seg_name in segments:
                    R_segment_data[seg_name].append(np.full((3,3), np.nan)) # Store NaN matrix
                for joint_name in joints:
                    R_joint_data[joint_name].append(np.full((3,3), np.nan)) # Store NaN matrix
                    angle_data[joint_name].append(np.full(3, np.nan)) # Store NaN angles
                frame_numbers.append(frame_count) # Still record frame number even if angles are NaN

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

# --- Convert angle lists to NumPy arrays for easier plotting ---
for joint in joints:
    angle_data[joint] = np.array(angle_data[joint])

# --- Plotting the Cardan angles ---
if frame_numbers:
    plt.figure(figsize=(18, 15))

    # Hip Angles
    plt.subplot(3, 2, 1)
    plt.plot(frame_numbers, angle_data['R_hip'][:, 0], label='Right Hip Flexion/Extension (X)')
    plt.plot(frame_numbers, angle_data['R_hip'][:, 1], label='Right Hip Abduction/Adduction (Y)')
    #plt.plot(frame_numbers, angle_data['R_hip'][:, 2], label='Right Hip Internal/External Rotation (Z)')
    plt.xlabel('Frame')
    plt.ylabel('Angle (degrees)')
    plt.title('Right Hip Cardan Angles')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(frame_numbers, angle_data['L_hip'][:, 0], label='Left Hip Flexion/Extension (X)')
    plt.plot(frame_numbers, angle_data['L_hip'][:, 1], label='Left Hip Abduction/Adduction (Y)')
    #plt.plot(frame_numbers, angle_data['L_hip'][:, 2], label='Left Hip Internal/External Rotation (Z)')
    plt.xlabel('Frame')
    plt.ylabel('Angle (degrees)')
    plt.title('Left Hip Cardan Angles')
    plt.grid(True)
    plt.legend()

    # Knee Angles
    plt.subplot(3, 2, 3)
    plt.plot(frame_numbers, angle_data['R_knee'][:, 0], label='Right Knee Flexion/Extension (X)')
    plt.plot(frame_numbers, angle_data['R_knee'][:, 1], label='Right Knee Abduction/Adduction (Y)')
    #plt.plot(frame_numbers, angle_data['R_knee'][:, 2], label='Right Knee Internal/External Rotation (Z)')
    plt.xlabel('Frame')
    plt.ylabel('Angle (degrees)')
    plt.title('Right Knee Cardan Angles')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(frame_numbers, angle_data['L_knee'][:, 0], label='Left Knee Flexion/Extension (X)')
    plt.plot(frame_numbers, angle_data['L_knee'][:, 1], label='Left Knee Abduction/Adduction (Y)')
    #plt.plot(frame_numbers, angle_data['L_knee'][:, 2], label='Left Knee Internal/External Rotation (Z)')
    plt.xlabel('Frame')
    plt.ylabel('Angle (degrees)')
    plt.title('Left Knee Cardan Angles')
    plt.grid(True)
    plt.legend()

    # Ankle Angles
    plt.subplot(3, 2, 5)
    plt.plot(frame_numbers, angle_data['R_ankle'][:, 0], label='Right Ankle Dorsi/Plantarflexion (X)')
    plt.plot(frame_numbers, angle_data['R_ankle'][:, 1], label='Right Ankle Eversion/Inversion (Y)')
    #plt.plot(frame_numbers, angle_data['R_ankle'][:, 2], label='Right Ankle Abduction/Adduction (Z)')
    plt.xlabel('Frame')
    plt.ylabel('Angle (degrees)')
    plt.title('Right Ankle Cardan Angles')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.plot(frame_numbers, angle_data['L_ankle'][:, 0], label='Left Ankle Dorsi/Plantarflexion (X)')
    plt.plot(frame_numbers, angle_data['L_ankle'][:, 1], label='Left Ankle Eversion/Inversion (Y)')
    #plt.plot(frame_numbers, angle_data['L_ankle'][:, 2], label='Left Ankle Abduction/Adduction (Z)')
    plt.xlabel('Frame')
    plt.ylabel('Angle (degrees)')
    plt.title('Left Ankle Cardan Angles')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Save to CSV
    df_data = {'Frame': frame_numbers}
    for joint in joints:
        df_data[f'{joint}_X'] = angle_data[joint][:, 0]
        df_data[f'{joint}_Y'] = angle_data[joint][:, 1]
        df_data[f'{joint}_Z'] = angle_data[joint][:, 2]

    df = pd.DataFrame(df_data)
    df.to_csv('mediapipe_cardan_angles.csv', index=False)
    print("MediaPipe Cardan angles saved to mediapipe_cardan_angles.csv")

else:
    print("No pose data found to plot.")
