import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import math # Used for asin, atan2 etc. directly

# Initialize MediaPipe Pose outside the function to leverage caching,
# but ensure it's available globally within this module.
# In a production environment, you might load this once in the FastAPI app startup.
mp_pose = mp.solutions.pose

# --- Helper Functions ---

def get_landmark_coords(landmarks, landmark_enum):
    """
    Helper to get 3D coords from MediaPipe landmark object.
    Returns a 3-element NumPy array, or full(3, NaN) if landmark is missing or invalid.
    MediaPipe world landmarks are in meters, origin typically near mid-hips for lower body.
    """
    if landmarks and landmark_enum.value < len(landmarks):
        lm = landmarks[landmark_enum.value]
        if lm and hasattr(lm, 'x') and hasattr(lm, 'y') and hasattr(lm, 'z'):
            # Ensure coordinates are not NaN, though MediaPipe usually provides values
            if np.isnan(lm.x) or np.isnan(lm.y) or np.isnan(lm.z):
                return np.full(3, np.nan)
            return np.array([lm.x, lm.y, lm.z])
    return np.full(3, np.nan) # Return NaN array if landmark is missing or incomplete

def cardanangles(R3):
    """
    Compute Cardan (XYZ) angles from a 3x3 rotation matrix R3 in radians.
    Based on your provided MATLAB-like conventions (Rx about X, Ry about Y, Rz about Z).

    Args:
        R3 (np.array): A 3x3 rotation matrix.

    Returns:
        np.array: A 3-element array of angles [Rx, Ry, Rz] in radians.
                  Returns NaNs if input is invalid or leads to gimbal lock issues.
    """
    if R3.shape != (3, 3):
        return np.full(3, np.nan) # Return NaN array for incorrect shape

    # Extract relevant sine/cosine terms for X, Y, Z intrinsic rotations
    # R = Rz(Rz) * Ry(Ry) * Rx(Rx)
    # R = [CyCz - SzSySx, CySz + CzSySx, -SyCx]
    #     [-CzSx      , CzCx      , Sx    ]
    #     [CySx + SzSyCx, SySz - CzSyCx, CyCx  ]
    Sx = R3[1, 2] # Corrected based on common Cardan XYZ definition (R3[2,1] from your snippet does not match the matrix structure provided)
    Cx = R3[2, 2] # Corrected
    Sy = -R3[0, 2]
    Cy = np.sqrt(1 - Sy**2) # Needs careful handling for atan2 later
    Sz = R3[0, 1]
    Cz = R3[0, 0]


    # Clamp Sx for numerical stability to avoid issues with arcsin outside [-1, 1]
    Sx_clamped = np.clip(Sx, -1.0, 1.0)
    Rx = np.arcsin(Sx_clamped) # Rotation about X (Pitch)

    # Gimbal lock check for pitch (Rx) near +/- 90 degrees
    # If cos(Rx) is near zero, then CzCx and SzCx will be near zero, making Rz and Ry indeterminate.
    # We use a small epsilon for float comparisons.
    if np.abs(np.cos(Rx)) < 1e-6: # Gimbal lock (Rx is +/- 90 degrees)
        # In gimbal lock, Rz and Ry are coupled. A common convention is to set one to zero
        # or calculate the sum/difference. Here, we'll calculate the sum (Ry + Rz)
        # or differences (Ry - Rz) based on the specific matrix.
        # For this setup, if Cx is near zero, we solve for (Ry + Rz)
        Ry = np.arctan2(R3[0, 1], R3[0, 0]) # This will be (Rz + Ry)
        Rz = 0.0 # Set Rz to 0 as it becomes arbitrary
    else:
        Ry = np.arctan2(Sy, R3[2, 2] / np.cos(Rx)) # Rotation about Y (Yaw)
        Rz = np.arctan2(-R3[1, 0], R3[1, 1]) # Rotation about Z (Roll)

    return np.array([Rx, Ry, Rz])


def safe_normalize(vec):
    """Normalizes a vector, returns NaN array if norm is zero."""
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-6 else np.full_like(vec, np.nan) # Use a small epsilon for comparison

def build_segment_rotation_matrix(proximal_lm, distal_lm, ref_lm1=None, ref_lm2=None, segment_type=''):
    """
    Builds a 3x3 rotation matrix for a segment's local coordinate system (SCS)
    relative to the MediaPipe world coordinate system.

    Args:
        proximal_lm (np.array): 3D coordinates of the proximal landmark.
        distal_lm (np.array): 3D coordinates of the distal landmark.
        ref_lm1 (np.array, optional): A third landmark for axis definition (e.g., R_shoulder for pelvis).
        ref_lm2 (np.array, optional): A fourth landmark for axis definition (e.g., L_shoulder for pelvis).
        segment_type (str): 'pelvis', 'thigh', 'shank' for specific axis definitions.

    Returns:
        np.array: A 3x3 rotation matrix (columns are X, Y, Z axes of the segment).
                  Returns a 3x3 NaN matrix if calculation fails due to missing/invalid landmarks.
    """
    # Check for NaN inputs for essential points right at the start
    if np.any(np.isnan(proximal_lm)) or np.any(np.isnan(distal_lm)):
        return np.full((3,3), np.nan)
    if segment_type == 'pelvis' and (np.any(np.isnan(ref_lm1)) or np.any(np.isnan(ref_lm2))):
        return np.full((3,3), np.nan)

    X_axis = np.full(3, np.nan)
    Y_axis = np.full(3, np.nan)
    Z_axis = np.full(3, np.nan)

    if segment_type == 'pelvis':
        # Pelvis SCS using R_HIP, L_HIP, R_SHOULDER, L_SHOULDER
        # This is a common biomechanical convention:
        # X-axis (Flexion/Extension): pointing anteriorly (forward)
        # Y-axis (Abduction/Adduction): pointing to subject's left (medio-lateral)
        # Z-axis (Rotation): pointing superiorly (up)

        R_HIP = proximal_lm
        L_HIP = distal_lm
        R_SHOULDER = ref_lm1
        L_SHOULDER = ref_lm2

        # Define the Medio-Lateral (Y) axis as vector from R_HIP to L_HIP
        Y_axis = safe_normalize(L_HIP - R_HIP)

        # Define a vector roughly representing the superior direction: mid-hip to mid-shoulder
        mid_hip = (R_HIP + L_HIP) / 2
        mid_shoulder = (R_SHOULDER + L_SHOULDER) / 2
        approx_superior_vec = mid_shoulder - mid_hip
        approx_superior_vec = safe_normalize(approx_superior_vec)

        # Z-axis (Superior-Inferior): cross product of X_approx and Y_axis
        # To get Z_axis (superior), we need X_axis (anterior).
        # A common approach for Z is perpendicular to the plane formed by hips and shoulders.
        # Let's derive Z from a plane and then X from Z and Y.
        # This method is more robust for defining Z: Z is the normal to the plane defined by:
        # R_HIP, L_HIP, and R_SHOULDER.
        vec1 = L_HIP - R_HIP
        vec2 = R_SHOULDER - R_HIP
        # Z-axis (Superior-Inferior): Cross product of R_HIP->L_HIP and R_HIP->R_SHOULDER
        # This gives a normal to the hip-shoulder plane. Conventionally, this is Z (upwards).
        Z_axis = safe_normalize(np.cross(vec1, vec2))
        # Ensure Z points generally "up" relative to MediaPipe's Y-down world.
        # If it's pointing down, flip it.
        if Z_axis[1] > 0: # MediaPipe Y is downwards, so positive Y means downwards. We want Z up.
            Z_axis *= -1

        # X-axis (Anterior-Posterior): cross product of Y and Z
        X_axis = safe_normalize(np.cross(Y_axis, Z_axis))

        # Re-orthogonalize Y-axis to ensure perfect orthogonality
        Y_axis = safe_normalize(np.cross(Z_axis, X_axis))

        # Final check for NaNs after all calculations
        if np.any(np.isnan(X_axis)) or np.any(np.isnan(Y_axis)) or np.any(np.isnan(Z_axis)):
            return np.full((3,3), np.nan)

    elif segment_type in ['thigh', 'shank']:
        # For thigh/shank:
        # Z-axis (Longitudinal): from distal to proximal (e.g., knee to hip for thigh)
        # X-axis (Flexion/Extension): roughly medio-lateral (perpendicular to sagittal plane)
        # Y-axis (Abduction/Adduction): roughly anterior-posterior

        # Z-axis: from distal (e.g., knee) to proximal (e.g., hip)
        Z_axis = safe_normalize(proximal_lm - distal_lm)

        # X-axis: Define as the axis perpendicular to the segment (Z_axis) and the global vertical (MediaPipe Y-down).
        # This approximates the flexion/extension axis of the knee/hip.
        global_vertical = np.array([0., -1., 0.]) # MediaPipe's Y-down, so -Y is upwards
        X_axis = safe_normalize(np.cross(Z_axis, global_vertical))

        # Fallback if the segment is perfectly vertical (cross product with global_vertical is zero)
        if np.any(np.isnan(X_axis)) or np.linalg.norm(X_axis) < 1e-6:
             # If Z is vertical, cross with global X to get a perpendicular vector
             X_axis = safe_normalize(np.cross(Z_axis, np.array([1., 0., 0.]))) # Global X-axis

        # Y-axis (Anterior-Posterior): Completes the right-handed system (cross product of Z and X)
        Y_axis = safe_normalize(np.cross(Z_axis, X_axis))

        # Re-orthogonalize X-axis to ensure perfect orthogonality
        X_axis = safe_normalize(np.cross(Y_axis, Z_axis))

        if np.any(np.isnan(X_axis)) or np.any(np.isnan(Y_axis)) or np.any(np.isnan(Z_axis)):
            return np.full((3,3), np.nan)

    else: # Unknown segment type
        return np.full((3,3), np.nan)

    # Return rotation matrix where columns are the basis vectors (X, Y, Z)
    return np.column_stack((X_axis, Y_axis, Z_axis))


# --- Main Angle Calculation Function ---
def get_mediapipe_angles(video_path: str) -> np.array:
    """
    Calculates Cardan (XYZ) angles for pelvis, hips, and knees from a video file
    using MediaPipe Pose, and returns them as a NumPy array.

    Args:
        video_path (str): The path to the input video file.

    Returns:
        np.array: A NumPy array containing the calculated angles for each frame.
                  Columns ordered as:
                  'pelvis_X', 'pelvis_Y',
                  'L_knee_X', 'L_knee_Y', 'L_knee_Z',
                  'R_knee_X', 'R_knee_Y', 'R_knee_Z',
                  'L_hip_X', 'L_hip_Y', 'L_hip_Z',
                  'R_hip_X', 'R_hip_Y', 'R_hip_Z'
                  Returns an empty array if no data is processed or video fails.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return np.array([]) # Return empty array if video fails to open

    # Define the desired column order for the output DataFrame.
    # This MUST match the input order your RNN expects.
    # Note: Adding Z angles for hip/knee for completeness based on standard Cardan angles.
    final_columns_order = [
        'Frame', # Keeping frame for debugging/tracking, will drop before returning final array
        'pelvis_X', 'pelvis_Y', 'pelvis_Z', # Pelvis angles (e.g., Flexion, Abduction, Rotation)
        'L_knee_X', 'L_knee_Y', 'L_knee_Z', # L_knee angles (Flexion, Abduction, Rotation)
        'R_knee_X', 'R_knee_Y', 'R_knee_Z', # R_knee angles
        'L_hip_X', 'L_hip_Y', 'L_hip_Z', # L_hip angles
        'R_hip_X', 'R_hip_Y', 'R_hip_Z', # R_hip angles
    ]

    angle_data_list = [] # List to store angle dictionaries for each frame

    # Initialize MediaPipe Pose model. This will be re-initialized for each call,
    # which might be slow. For a long-running service, consider loading once at app startup.
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2, # Higher complexity for better accuracy, slower
        enable_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Convert to RGB as MediaPipe expects RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False # To improve performance
        results = pose.process(image)
        image.flags.writeable = True

        # Initialize a dictionary for the current frame's angles, filled with NaNs
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

    cap.release()
    pose.close() # Important to close MediaPipe pose object to release resources

    if not angle_data_list:
        print("No pose data found for any frame. Returning empty array.")
        return np.array([])

    # Convert collected angle data to a Pandas DataFrame and enforce column order
    final_angles_df = pd.DataFrame(angle_data_list)

    # Ensure all required columns exist, filling with NaN if a column is entirely missing
    for col in final_columns_order:
        if col not in final_angles_df.columns:
            final_angles_df[col] = np.nan

    # Reindex the DataFrame to ensure the desired column order (important!)
    final_angles_df = final_angles_df[final_columns_order]

    # Convert DataFrame to NumPy array for RNN input
    # Drop 'Frame' column as it's typically not a feature for the RNN
    return final_angles_df.drop(columns=['Frame']).values
