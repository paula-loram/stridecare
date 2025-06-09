import cv2
import mediapipe as mp
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.transform import Rotation as R_scipy # Renamed to avoid conflict

# The aim of this notebook is to calculate the angles over frame:
# L/R Knee (x & y)
# L/R Hip (x & y)
# pelvis_global (x & y)

def cardanangles(R3):
    """
    Compute Cardan (XYZ) angles from a 3×3 rotation matrix R3.
    Returns [Rx, Ry, Rz] in radians, following the same conventions
    as your MATLAB `cardanangles`:

      | Cz*Cy - Sz*Sy*Sx   Sz*Cy + Cz*Sy*Sx  -Sy*Cx |
      | -Sz*Cx             Cz*Cx              Sx     |
      | Cz*Sy + Cz*Sy*Sx   Sz*Sy - Cz*Cy*Sx  Cy*Cx  |

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
        Rx = np.arctan2(R3[2, 1], R3[1, 1])
        Ry = np.arctan2(-R3[0, 2], R3[2, 2])
        Rz = np.arctan2(-R3[1, 0], R3[1, 1])

    return np.array([Rx, Ry, Rz])

# --- Helper function for MediaPipe Landmark Coordinates ---
def get_landmark_coords(landmarks, landmark_enum):
    """Helper to get 3D coords from landmark object, returns NaN array if missing."""
    if landmarks and landmark_enum.value < len(landmarks):
        lm = landmarks[landmark_enum.value]
        # MediaPipe world landmarks are in meters, origin at mid-hip for lower body
        if lm and hasattr(lm, 'x') and hasattr(lm, 'y') and hasattr(lm, 'z'):
            return np.array([lm.x, lm.y, lm.z])
    return np.full(3, np.nan) # Return NaN array if landmark is missing or incomplete

# --- Helper function for safe normalization ---
def safe_normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else np.full_like(vec, np.nan)

# --- EDITED FUNCTION FOR MEDIAPIPE ANGLES ---
def build_segment_rotation_matrix(proximal_lm, distal_lm, ref_lm_for_x=None, ref_lm_for_y=None, segment_type=''):
    """
    Builds a 3x3 rotation matrix for a segment's local coordinate system (SCS)
    relative to the MediaPipe world coordinate system.

    Args:
        proximal_lm (np.array): 3D coordinates of the proximal landmark.
        distal_lm (np.array): 3D coordinates of the distal landmark.
        ref_lm_for_x (np.array, optional): A third landmark for axis definition (e.g., R_shoulder for pelvis, Foot_index for foot).
        ref_lm_for_y (np.array, optional): A fourth landmark for axis definition (e.g., L_shoulder for pelvis, Heel for foot).
        segment_type (str): 'pelvis', 'thigh', 'shank', 'foot' for specific axis definitions.

    Returns:
        np.array: A 3x3 rotation matrix (columns are X, Y, Z axes of the segment).
                  Returns a 3x3 NaN matrix if calculation fails due to missing/invalid landmarks.
    """
    # Initialize axes as NaNs
    X_axis = np.full(3, np.nan)
    Y_axis = np.full(3, np.nan)
    Z_axis = np.full(3, np.nan)

    # Check for NaN inputs for essential points
    if np.any(np.isnan(proximal_lm)) or np.any(np.isnan(distal_lm)):
        return np.full((3,3), np.nan)

    if segment_type == 'pelvis':
        # Pelvis SCS using R_HIP, L_HIP, R_SHOULDER, L_SHOULDER
        # This aligns with your training data's primary pelvis definition.
        # proximal_lm = R_HIP, distal_lm = L_HIP
        # ref_lm_for_x = R_SHOULDER, ref_lm_for_y = L_SHOULDER

        if (ref_lm_for_x is not None and not np.any(np.isnan(ref_lm_for_x)) and
            ref_lm_for_y is not None and not np.any(np.isnan(ref_lm_for_y))):

            mid_hip = (proximal_lm + distal_lm) / 2
            mid_shoulder = (ref_lm_for_x + ref_lm_for_y) / 2

            # Z-axis (Longitudinal/Superior-Inferior): From mid-hip to mid-shoulder (pointing upwards)
            Z_axis = safe_normalize(mid_shoulder - mid_hip)

            # Y-axis (Medio-Lateral): From R_HIP to L_HIP (points to subject's left)
            Y_axis = safe_normalize(distal_lm - proximal_lm) # L_HIP - R_HIP

            # X-axis (Anterior-Posterior): Cross product of Y and Z (points roughly forward)
            X_axis = safe_normalize(np.cross(Y_axis, Z_axis))

            # Re-orthogonalize Y_axis to ensure perfect orthogonality
            Y_axis = safe_normalize(np.cross(Z_axis, X_axis))

            # Re-check for NaNs after normalization/cross product
            if np.any(np.isnan(X_axis)) or np.any(np.isnan(Y_axis)) or np.any(np.isnan(Z_axis)):
                print(f"Pelvis: Axes became NaN after calculation, falling back.")
                # Fallback to global alignment if calculation failed with specific landmarks
                # MediaPipe's global frame is: +X to subject's right, +Y downwards, +Z into the camera.
                # Pelvis Z (longitudinal/vertical): opposite MediaPipe Y (so positive up)
                Z_axis = np.array([0., -1., 0.])
                # Pelvis X (medio-lateral): opposite MediaPipe X (so points L_hip to R_hip for consistency)
                X_axis = np.array([-1., 0., 0.]) # Points from Left to Right (subject's left is MP's -X)
                # Pelvis Y (anterior-posterior): cross product of Z and X
                Y_axis = safe_normalize(np.cross(Z_axis, X_axis))
                # Re-orthogonalize X_axis
                X_axis = safe_normalize(np.cross(Y_axis, Z_axis))


        else: # Fallback for pelvis if shoulders are missing or invalid in MediaPipe
            # Align pelvis axes to a common biomechanical setup relative to MediaPipe's global frame.
            # MediaPipe's global frame is: +X to subject's right, +Y downwards, +Z into the camera.
            # We want: X (anterior-posterior), Y (medio-lateral), Z (superior-inferior)
            # Pelvis Z (longitudinal/vertical): opposite MediaPipe Y (so positive up)
            Z_axis = np.array([0., -1., 0.])
            # Pelvis Y (medio-lateral): opposite MediaPipe X (so points L_hip to R_hip)
            Y_axis = np.array([-1., 0., 0.]) # Points from Left to Right (subject's left is MP's -X)
            # Pelvis X (anterior-posterior): cross product of Y and Z (points roughly forward, towards camera)
            X_axis = safe_normalize(np.cross(Y_axis, Z_axis))
            # Re-orthogonalize Y_axis
            Y_axis = safe_normalize(np.cross(Z_axis, X_axis))

    elif segment_type in ['thigh', 'shank']:
        # Z-axis (Longitudinal): Vector from distal to proximal landmark
        # For thigh: knee to hip. For shank: ankle to knee.
        # This ensures the segment's Z-axis points proximally.
        Z_axis = safe_normalize(proximal_lm - distal_lm)

        # X-axis (Medio-Lateral): Attempt to define this as the flexion/extension axis
        # Use a global vertical vector (opposite MediaPipe's +Y is up)
        global_up = np.array([0., -1., 0.])
        X_axis = safe_normalize(np.cross(Z_axis, global_up)) # Perpendicular to longitudinal and vertical

        # Fallback if the segment is perfectly vertical (cross product with global_up is zero)
        if np.any(np.isnan(X_axis)) or np.linalg.norm(X_axis) < 1e-6:
             # Cross with global X (right) to get a different perpendicular vector
             X_axis = safe_normalize(np.cross(Z_axis, np.array([1., 0., 0.])))

        # Y-axis (Anterior-Posterior): cross product of Z and X
        Y_axis = safe_normalize(np.cross(Z_axis, X_axis))

        # Re-orthogonalize X_axis for perfect rotation matrix (X, Y, Z must be orthogonal)
        X_axis = safe_normalize(np.cross(Y_axis, Z_axis))

    elif segment_type == 'foot':
        # This segment type is now REMOVED from calculation as per previous request.
        # However, if you were to re-enable it and want it to be *comparable*
        # to your training data's foot definition, this is a highly
        # challenging area due to different marker sets.
        #
        # Your training data used 'L_foot_1', 'L_foot_2', 'L_foot_3', 'L_foot_4'
        # with a specific sorting and cross-product logic.
        # MediaPipe provides 'ANKLE', 'HEEL', 'FOOT_INDEX' (toe).
        #
        # A *highly speculative* attempt to mimic your training data's foot logic
        # with MediaPipe points for general understanding:
        # Assuming distal_lm is HEEL, proximal_lm is ANKLE (unused for foot in your case)
        # ref_lm_for_x is FOOT_INDEX (toe)
        # ref_lm_for_y is (could be ball of foot if available, or midpoint)

        # If we had to define a foot segment from MediaPipe landmarks:
        # Z-axis (Longitudinal): from Heel to Foot_index (toe)
        # if ref_lm_for_x is not None and not np.any(np.isnan(ref_lm_for_x)) and \
        #    distal_lm is not None and not np.any(np.isnan(distal_lm)):
        #     Z_axis = safe_normalize(ref_lm_for_x - distal_lm) # Foot_index (toe) - Heel
        # else:
        #     return np.full((3,3), np.nan) # Cannot define foot SCS without toe/heel

        # X-axis (Medio-Lateral): Perpendicular to Z and roughly vertical
        # global_up = np.array([0., -1., 0.]) # MediaPipe's Y-down, so -Y is up
        # X_axis = safe_normalize(np.cross(Z_axis, global_up))

        # # Fallback for X-axis if Z is vertical (e.g., standing still)
        # if np.any(np.isnan(X_axis)) or np.linalg.norm(X_axis) < 1e-6:
        #     X_axis = safe_normalize(np.cross(Z_axis, np.array([1., 0., 0.]))) # Try global X

        # Y_axis (Vertical/Perpendicular to X-Z plane):
        # Y_axis = safe_normalize(np.cross(Z_axis, X_axis))
        # X_axis = safe_normalize(np.cross(Y_axis, Z_axis)) # Re-orthogonalize X

        # Since foot and ankle calculations are currently disabled, this block will not execute.
        return np.full((3,3), np.nan) # Default return for unhandled segments.

    else:
        # Should not reach here if segment_type is properly handled
        return np.full((3,3), np.nan)

    # Final check for NaNs before returning the matrix
    if np.any(np.isnan(X_axis)) or np.any(np.isnan(Y_axis)) or np.any(np.isnan(Z_axis)):
        return np.full((3,3), np.nan)

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
# Removed 'L_foot', 'R_foot' from segments as ankles are removed
segments = ['L_shank', 'R_shank', 'L_thigh', 'R_thigh', 'pelvis']
# Removed 'L_ankle', 'R_ankle' from joints as requested
# Added 'pelvis_global' to explicitly store its orientation
joints = ['L_knee', 'R_knee', 'L_hip', 'R_hip', 'pelvis_global']

# Initialize dictionaries to store angles for each joint
angle_data_df = [] # List of dictionaries to build DataFrame
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

        current_frame_angles = {'Frame': frame_count}

        if results.pose_landmarks and results.pose_world_landmarks:
            landmarks = results.pose_world_landmarks.landmark

            # Extract MediaPipe Landmarks - ensure all are extracted even if not immediately used by segment
            # This helps in debugging and understanding data availability
            r_hip = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
            r_knee = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE)
            r_ankle = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE)
            r_heel = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_HEEL)
            r_foot_index = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)
            r_shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)

            l_hip = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
            l_knee = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_KNEE)
            l_ankle = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
            l_heel = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_HEEL)
            l_foot_index = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_FOOT_INDEX)
            l_shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)

            current_R_segs = {}
            try:
                # Pelvis Rotation Matrix - uses R_HIP, L_HIP, R_SHOULDER, L_SHOULDER
                current_R_segs['pelvis'] = build_segment_rotation_matrix(
                    r_hip, l_hip, r_shoulder, l_shoulder, segment_type='pelvis'
                )

                # Right Thigh Rotation Matrix - uses R_HIP, R_KNEE
                current_R_segs['R_thigh'] = build_segment_rotation_matrix(
                    r_hip, r_knee, segment_type='thigh'
                )
                # Left Thigh Rotation Matrix - uses L_HIP, L_KNEE
                current_R_segs['L_thigh'] = build_segment_rotation_matrix(
                    l_hip, l_knee, segment_type='thigh'
                )

                # Right Shank Rotation Matrix - uses R_KNEE, R_ANKLE
                current_R_segs['R_shank'] = build_segment_rotation_matrix(
                    r_knee, r_ankle, segment_type='shank'
                )
                # Left Shank Rotation Matrix - uses L_KNEE, L_ANKLE
                current_R_segs['L_shank'] = build_segment_rotation_matrix(
                    l_knee, l_ankle, segment_type='shank'
                )

                # Pelvis Global Angles (Pelvis SCS relative to MediaPipe World)
                R_pelvis_global = current_R_segs.get('pelvis')
                if R_pelvis_global is not None and not np.any(np.isnan(R_pelvis_global)):
                    pelvis_angles_rad = cardanangles(R_pelvis_global)
                    pelvis_angles_deg = np.degrees(pelvis_angles_rad)
                    current_frame_angles[f'pelvis_global_X'] = pelvis_angles_deg[0]
                    current_frame_angles[f'pelvis_global_Y'] = pelvis_angles_deg[1]
                    # current_frame_angles[f'pelvis_global_Z'] = pelvis_angles_deg[2] # Z-angle commented out
                else:
                    current_frame_angles[f'pelvis_global_X'] = np.nan
                    current_frame_angles[f'pelvis_global_Y'] = np.nan
                    # current_frame_angles[f'pelvis_global_Z'] = np.nan # Z-angle commented out


                # L_hip: Pelvis (proximal) to Thigh (distal)
                if not np.any(np.isnan(current_R_segs.get('pelvis'))) and not np.any(np.isnan(current_R_segs.get('L_thigh'))):
                    R_L_hip = current_R_segs['pelvis'].T @ current_R_segs['L_thigh']
                    L_hip_angles = np.degrees(cardanangles(R_L_hip))
                    current_frame_angles[f'L_hip_X'] = L_hip_angles[0]
                    current_frame_angles[f'L_hip_Y'] = L_hip_angles[1]
                    # current_frame_angles[f'L_hip_Z'] = L_hip_angles[2] # Z-angle commented out
                else:
                    current_frame_angles[f'L_hip_X'] = np.nan
                    current_frame_angles[f'L_hip_Y'] = np.nan
                    # current_frame_angles[f'L_hip_Z'] = np.nan # Z-angle commented out

                # R_hip: Pelvis (proximal) to Thigh (distal)
                if not np.any(np.isnan(current_R_segs.get('pelvis'))) and not np.any(np.isnan(current_R_segs.get('R_thigh'))):
                    R_R_hip = current_R_segs['pelvis'].T @ current_R_segs['R_thigh']
                    R_hip_angles = np.degrees(cardanangles(R_R_hip))
                    current_frame_angles[f'R_hip_X'] = R_hip_angles[0]
                    current_frame_angles[f'R_hip_Y'] = R_hip_angles[1]
                    # current_frame_angles[f'R_hip_Z'] = R_hip_angles[2] # Z-angle commented out
                else:
                    current_frame_angles[f'R_hip_X'] = np.nan
                    current_frame_angles[f'R_hip_Y'] = np.nan
                    # current_frame_angles[f'R_hip_Z'] = np.nan # Z-angle commented out

                # L_knee: Thigh (proximal) to Shank (distal)
                if not np.any(np.isnan(current_R_segs.get('L_thigh'))) and not np.any(np.isnan(current_R_segs.get('L_shank'))):
                    R_L_knee = current_R_segs['L_thigh'].T @ current_R_segs['L_shank']
                    L_knee_angles = np.degrees(cardanangles(R_L_knee))
                    current_frame_angles[f'L_knee_X'] = L_knee_angles[0]
                    current_frame_angles[f'L_knee_Y'] = L_knee_angles[1]
                    # current_frame_angles[f'L_knee_Z'] = L_knee_angles[2] # Z-angle commented out
                else:
                    current_frame_angles[f'L_knee_X'] = np.nan
                    current_frame_angles[f'L_knee_Y'] = np.nan
                    # current_frame_angles[f'L_knee_Z'] = np.nan # Z-angle commented out

                # R_knee: Thigh (proximal) to Shank (distal)
                if not np.any(np.isnan(current_R_segs.get('R_thigh'))) and not np.any(np.isnan(current_R_segs.get('R_shank'))):
                    R_R_knee = current_R_segs['R_thigh'].T @ current_R_segs['R_shank']
                    R_knee_angles = np.degrees(cardanangles(R_R_knee))
                    current_frame_angles[f'R_knee_X'] = R_knee_angles[0]
                    current_frame_angles[f'R_knee_Y'] = R_knee_angles[1]
                    # current_frame_angles[f'R_knee_Z'] = R_knee_angles[2] # Z-angle commented out
                else:
                    current_frame_angles[f'R_knee_X'] = np.nan
                    current_frame_angles[f'R_knee_Y'] = np.nan
                    # current_frame_angles[f'R_knee_Z'] = np.nan # Z-angle commented out

            except Exception as e:
                # print(f"Could not compute segment or joint rotation for frame {frame_count}: {e}")
                # Fill current frame angles with NaNs if an error occurs
                for joint_col in joints:
                    # Check if keys exist from previous calculations to avoid KeyError
                    # If not, assume all are NaN for this frame's joint calculations
                    if f'{joint_col}_X' not in current_frame_angles: current_frame_angles[f'{joint_col}_X'] = np.nan
                    if f'{joint_col}_Y' not in current_frame_angles: current_frame_angles[f'{joint_col}_Y'] = np.nan
                    # if f'{joint_col}_Z' not in current_frame_angles: current_frame_angles[f'{joint_col}_Z'] = np.nan # Z-angle commented out

        else: # No landmarks detected for this frame
            # Fill current frame angles with NaNs for all relevant joints
            for joint_col in joints:
                current_frame_angles[f'{joint_col}_X'] = np.nan
                current_frame_angles[f'{joint_col}_Y'] = np.nan
                # current_frame_angles[f'{joint_col}_Z'] = np.nan # Z-angle commented out

        angle_data_df.append(current_frame_angles) # Append dictionary for this frame

        # Draw the pose annotation on the image. (Commented out as requested)
        # mp_drawing.draw_landmarks(
        #     image,
        #     results.pose_landmarks,
        #     mp_pose.POSE_CONNECTIONS,
        #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        #
        # cv2.imshow('MediaPipe Pose - Cardan Angles', image)
        #
        # if cv2.waitKey(5) & 0xFF == 27:
        #     break

cap.release()
cv2.destroyAllWindows()

# --- Convert collected angle data to a Pandas DataFrame ---
final_angles_df = pd.DataFrame(angle_data_df)

# --- Save to CSV ---
if not final_angles_df.empty:
    output_csv_path = 'mediapipe_cardan_angles_hip_knee_pelvis_no_z.csv'
    final_angles_df.to_csv(output_csv_path, index=False)
    print(f"MediaPipe Cardan angles saved to {output_csv_path}")
else:
    print("No pose data found and no angles computed.")
