import json
import numpy as np
import math
import pandas as pd
from scipy.spatial.transform import Rotation as R_scipy # Renamed to avoid conflict

# --- YOUR PROVIDED CARDAN ANGLES FUNCTION ---
def cardanangles(R3):
    """
    Compute Cardan (XYZ) angles from a 3×3 rotation matrix R3.
    Returns [Rx, Ry, Rz] in radians, following the same conventions
    as your MATLAB `cardanangles`:

      | Cz*Cy - Sz*Sy*Sx    Sz*Cy + Cz*Sy*Sx   -Sy*Cx |
      | -Sz*Cx              Cz*Cx              Sx     |
      | Cz*Sy + Cz*Sy*Sx    Sz*Sy - Cz*Cy*Sx   Cy*Cx  |

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

# Helper function to compute the rotation matrix from a set of source points to target points
def get_rotation_from_markers(source_points, target_points):
    """
    Calculates the rotation matrix from a set of source points to a set of target points.
    Points should be (N, 3) arrays, where N is the number of markers.
    """
    if source_points.shape[0] < 3 or target_points.shape[0] < 3:
        # Need at least 3 non-collinear points for a unique rotation
        return np.full((3,3), np.nan)

    # Center the points
    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)

    centered_source = source_points - centroid_source
    centered_target = target_points - centroid_target

    # Use scipy's align_vectors for robust rotation finding
    rotation, _ = R_scipy.align_vectors(centered_source, centered_target)
    return rotation.as_matrix()

# Helper function for safe normalization
def safe_normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else np.full_like(vec, np.nan)

# Helper function to compute segment anatomical-to-global transformation matrix (ag)
def compute_segment_ag(prox_jc, dist_jc, med_lm, lat_lm, segment_type, joints_data=None, neutral_data=None):
    if np.any(np.isnan(prox_jc)) or np.any(np.isnan(dist_jc)):
        return np.full((3,3), np.nan) # Return NaN matrix if JCs are missing

    X_axis = np.full(3, np.nan)
    Y_axis = np.full(3, np.nan)
    Z_axis = np.full(3, np.nan)

    if segment_type == 'foot':
        # Foot specific: Use neutral foot markers directly
        if neutral_data and all(marker in neutral_data for marker in ['_foot_1', '_foot_2', '_foot_3']): # Generic check
            foot_markers_arr = np.array([neutral_data[f'{segment_type}_1'], neutral_data[f'{segment_type}_2'], neutral_data[f'{segment_type}_3']])
            foot_markers_arr = foot_markers_arr[np.argsort(foot_markers_arr[:, 0])]

            # This is complex based on the original notebook. Re-implementing with more standard definition
            # Or simplified to be based on two key points to approximate axes
            # For simplicity for initial pass based on the original notebook's pattern:
            # Let's assume a Z-axis along the foot length (e.g. from heel to toe)
            # X-axis (medio-lateral) and Y-axis (vertical)

            # Original notebook's foot axis definition was quite specific and potentially fragile.
            # Sticking to its pattern for consistency for now, but be aware this might need refinement.
            x_ref = np.array([0, 0, -1]) # Arbitrary initial X to define Z
            temp_vec = safe_normalize(foot_markers_arr[1] - foot_markers_arr[2]) # Example from neutral logic
            if temp_vec[1] < 0: # Ensure consistent direction
                temp_vec = -temp_vec

            Z_axis = safe_normalize(np.cross(x_ref, temp_vec))
            Y_axis = safe_normalize(np.cross(Z_axis, x_ref))
            X_axis = safe_normalize(np.cross(Y_axis, Z_axis))
        else:
            print(f"Warning: Missing neutral foot markers for {segment_type} AG calculation.")
            return np.full((3,3), np.nan)

    elif segment_type in ['thigh', 'shank']:
        Y_axis = safe_normalize(prox_jc - dist_jc) # Longitudinal axis (proximal to distal)

        # Medio-lateral axis (X-axis) using epicondyles/malleoli if available
        if joints_data and med_lm in joints_data and lat_lm in joints_data:
            X_axis = safe_normalize(joints_data[med_lm] - joints_data[lat_lm])
        else:
            # Fallback for X-axis if specific anatomical markers are missing
            # A common approach is to use a "plane" normal or cross product with global up/forward
            # Example: Assuming global Y is vertical (down), cross with Y_axis to get a roughly horizontal X
            global_up = np.array([0, -1, 0]) # Assuming MediaPipe like Y-down convention, then Y-up would be [0, -1, 0]
            X_axis = safe_normalize(np.cross(Y_axis, global_up))
            if np.any(np.isnan(X_axis)): # If parallel to global_up, use another axis
                X_axis = safe_normalize(np.cross(Y_axis, np.array([1,0,0]))) # Try global X

        Z_axis = safe_normalize(np.cross(X_axis, Y_axis)) # Orthogonal to X and Y
        X_axis = safe_normalize(np.cross(Y_axis, Z_axis)) # Re-orthogonalize X for perfect rotation matrix

    else:
        # Pelvis needs special handling as it's not hip-to-knee or knee-to-ankle
        # The original notebook had a complex pelvis definition, let's keep it in the main function body
        # For simplicity, this helper only covers thigh/shank/foot as defined above
        return np.full((3,3), np.nan) # Should not reach here for pelvis if handled separately

    if np.any(np.isnan(X_axis)) or np.any(np.isnan(Y_axis)) or np.any(np.isnan(Z_axis)):
        return np.full((3,3), np.nan)

    return np.column_stack((X_axis, Y_axis, Z_axis))


def analyze_gait_data(filepath: str) -> pd.DataFrame:
    """
    Analyzes gait data from a specified JSON file to compute Cardan angles
    for Pelvis, Hip, Knee, and Ankle joints over time and returns them in a pandas DataFrame.
    The Z-angle (axial rotation) is excluded from the output DataFrame.

    Args:
        filepath (str): The path to the JSON data file.

    Returns:
        pd.DataFrame: A DataFrame containing Cardan angles (in degrees) for
                      Pelvis, Hip, Knee, and Ankle joints for each frame.
                      Returns an empty DataFrame if data loading or processing fails.
                      Columns: 'Frame', 'L_hip_X', 'L_hip_Y', 'L_knee_X', etc. (Z-angles excluded)
    """
    # --- Step 1: Configuration ---
    HZ_KEY = 'hz_r'
    STATIC_JOINTS_MARKERS_KEY = 'joints'
    STATIC_NEUTRAL_MARKERS_KEY = 'neutral'
    DYNAMIC_MARKERS_KEY = 'running'

    # Define tracking marker sets for each segment
    tracking_marker_sets = {
        'pelvis': ['pelvis_1', 'pelvis_2', 'pelvis_3', 'pelvis_4'],
        'L_thigh': ['L_thigh_1', 'L_thigh_2', 'L_thigh_3', 'L_thigh_4'],
        'R_thigh': ['R_thigh_1', 'R_thigh_2', 'R_thigh_3', 'R_thigh_4'],
        'L_shank': ['L_shank_1', 'L_shank_2', 'L_shank_3', 'L_shank_4'],
        'R_shank': ['R_shank_1', 'R_shank_2', 'R_shank_3', 'R_shank_4'],
        'L_foot': ['L_foot_1', 'L_foot_2', 'L_foot_3', 'L_foot_4', 'L_toe'],
        'R_foot': ['R_foot_1', 'R_foot_2', 'R_foot_3', 'R_foot_4', 'R_toe']
    }

    # --- Step 2: Load Data ---
    joints = None
    neutral = None
    dynamic_data_raw = {} # Raw dict to store marker data
    hz = None
    num_frames = 0

    # Store neutral marker positions as a (N, 3) array for each cluster
    neutral_cluster_markers_pos = {}

    # Store mapping from segment name to it's ag matrix from neutral pose
    ag_neutral = {}

    # Load static joint data (anatomical markers)
    try:
        with open(filepath, 'r') as f:
            static_joints_json_data = json.load(f)
        raw_joints_markers = static_joints_json_data.get(STATIC_JOINTS_MARKERS_KEY, {})
        joints = {name: np.array(data) for name, data in raw_joints_markers.items() if data is not None}
        if not all(m in joints for m in ['L_hip', 'R_hip', 'L_lat_knee', 'L_med_knee', 'R_lat_knee', 'R_med_knee', 'L_lat_ankle', 'L_med_ankle', 'R_lat_ankle', 'R_med_ankle']):
            print(f"Warning: Some required anatomical markers missing in {filepath}")

    except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
        print(f"Error loading static joints data from {filepath}: {e}")
        return pd.DataFrame()

    # Load static neutral data (tracking markers in neutral pose)
    try:
        with open(filepath, 'r') as f:
            static_neutral_json_data = json.load(f)
        raw_neutral_markers = static_neutral_json_data.get(STATIC_NEUTRAL_MARKERS_KEY, {})
        neutral = {name: np.array(data) for name, data in raw_neutral_markers.items() if data is not None}
        if not all(m in neutral for m_list in tracking_marker_sets.values() for m in m_list):
             print(f"Warning: Some required neutral tracking markers missing in {filepath}")

    except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
        print(f"Error loading static neutral data from {filepath}: {e}")
        return pd.DataFrame()

    # Load dynamic data (tracking markers time-series and sampling frequency)
    try:
        with open(filepath, 'r') as f:
            dynamic_json_data = json.load(f)
        hz = dynamic_json_data.get(HZ_KEY)
        if hz is None:
            print(f"Warning: Sampling frequency key '{HZ_KEY}' not found in {filepath}. Using default HZ=120.")
            hz = 120

        raw_dynamic_markers = dynamic_json_data.get(DYNAMIC_MARKERS_KEY, {})

        # Determine num_frames from the first available dynamic marker in tracking_marker_sets
        first_dynamic_marker = next((m for seg in tracking_marker_sets.values() for m in seg if m in raw_dynamic_markers and raw_dynamic_markers[m]), None)
        if first_dynamic_marker:
            num_frames = len(raw_dynamic_markers[first_dynamic_marker])
        else:
            print(f"Warning: No valid dynamic marker data to determine number of frames in {filepath}.")
            return pd.DataFrame()

        # Populate dynamic_data_raw and handle missing data with NaNs
        for marker_name in {m for seg in tracking_marker_sets.values() for m in seg}: # Use all markers in any cluster
            marker_data_series = raw_dynamic_markers.get(marker_name)
            if marker_data_series is not None and len(marker_data_series) == num_frames:
                dynamic_data_raw[marker_name] = np.array(marker_data_series)
            else:
                dynamic_data_raw[marker_name] = np.full((num_frames, 3), np.nan) # Fill with NaNs
                # print(f"Warning: Dynamic marker '{marker_name}' missing or inconsistent length in {filepath}")

    except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
        print(f"Error loading dynamic data from {filepath}: {e}")
        return pd.DataFrame()

    # --- Step 3: Verification ---
    if not (joints and neutral and dynamic_data_raw and hz is not None and num_frames > 0):
        print(f"Failed to load all required data for {filepath}. Kinematics analysis cannot proceed.")
        return pd.DataFrame()

    # --- Step 4: Compute Joint Centers (jc) ---
    jc = {}
    jc['pelvis'] = np.mean([neutral[m] for m in ['pelvis_1', 'pelvis_2', 'pelvis_3', 'pelvis_4'] if m in neutral], axis=0) if all(m in neutral for m in ['pelvis_1', 'pelvis_2', 'pelvis_3', 'pelvis_4']) else np.full(3, np.nan)
    jc['L_hip'] = joints.get('L_hip', np.full(3, np.nan)) + (joints.get('R_hip', np.full(3, np.nan)) - joints.get('L_hip', np.full(3, np.nan))) / 4
    jc['R_hip'] = joints.get('R_hip', np.full(3, np.nan)) + (joints.get('L_hip', np.full(3, np.nan)) - joints.get('R_hip', np.full(3, np.nan))) / 4
    jc['L_knee'] = np.mean([joints.get('L_lat_knee', np.full(3, np.nan)), joints.get('L_med_knee', np.full(3, np.nan))], axis=0)
    jc['R_knee'] = np.mean([joints.get('R_lat_knee', np.full(3, np.nan)), joints.get('R_med_knee', np.full(3, np.nan))], axis=0)
    jc['L_ankle'] = np.mean([joints.get('L_lat_ankle', np.full(3, np.nan)), joints.get('L_med_ankle', np.full(3, np.nan))], axis=0)
    jc['R_ankle'] = np.mean([joints.get('R_lat_ankle', np.full(3, np.nan)), joints.get('R_med_ankle', np.full(3, np.nan))], axis=0)

    # --- Step 5: Compute Anatomical-to-Global Transformation Matrices (ag) in neutral pose ---
    # These define the anatomical coordinate systems (ACS) in the global frame during the static neutral trial.
    # ag_neutral will map from segment ACS to Global frame in neutral pose.
    ag_neutral = {}

    # Pelvis (Special case, often defined differently)
    # Using the neutral markers and a common definition for pelvis axes
    pelvis_markers_neutral = np.array([neutral[m] for m in tracking_marker_sets['pelvis'] if m in neutral])
    if pelvis_markers_neutral.shape[0] >= 3:
        # A common pelvis definition (e.g., Z-axis vertical, X-axis between hips, Y-axis forward)
        # Using the approach from the original notebook (mid_hip, mid_shoulder if available)
        r_hip_n = neutral.get('R_hip') or jc['R_hip']
        l_hip_n = neutral.get('L_hip') or jc['L_hip']
        r_shoulder_n = neutral.get('R_shoulder')
        l_shoulder_n = neutral.get('L_shoulder')

        if r_hip_n is not None and l_hip_n is not None:
            mid_hip = (r_hip_n + l_hip_n) / 2
            if r_shoulder_n is not None and l_shoulder_n is not None:
                mid_shoulder = (r_shoulder_n + l_shoulder_n) / 2
                Z_axis_p = safe_normalize(mid_shoulder - mid_hip)
                Y_axis_p = safe_normalize(l_hip_n - r_hip_n) # Points from R to L
                X_axis_p = safe_normalize(np.cross(Y_axis_p, Z_axis_p))
                Y_axis_p = safe_normalize(np.cross(Z_axis_p, X_axis_p)) # Re-orthogonalize
            else: # Fallback without shoulder markers
                # Simple pelvis aligned to global X-Z plane, with Y in sagittal plane
                # Need a robust definition for pelvis in neutral if shoulders are missing from JSON
                # For now, will use simpler one and assume vertical Y axis (up) for torso
                Z_axis_p = np.array([0, 1, 0]) # Vertical (up)
                X_axis_p = safe_normalize(l_hip_n - r_hip_n) # Medio-lateral (L-R)
                Y_axis_p = safe_normalize(np.cross(Z_axis_p, X_axis_p))
                Z_axis_p = safe_normalize(np.cross(X_axis_p, Y_axis_p)) # Re-orthogonalize

            if not np.any(np.isnan(X_axis_p)) and not np.any(np.isnan(Y_axis_p)) and not np.any(np.isnan(Z_axis_p)):
                ag_neutral['pelvis'] = np.column_stack((X_axis_p, Y_axis_p, Z_axis_p))
            else:
                ag_neutral['pelvis'] = np.full((3,3), np.nan)
                print(f"Warning: Could not define pelvis AG for {filepath} due to NaN axes.")
        else:
            ag_neutral['pelvis'] = np.full((3,3), np.nan)
            print(f"Warning: Missing hip markers for pelvis AG calculation in {filepath}")
    else:
        ag_neutral['pelvis'] = np.full((3,3), np.nan)
        print(f"Warning: Not enough neutral pelvis markers for AG calculation in {filepath}")


    # Other segments (Thigh, Shank, Foot) using the helper
    ag_neutral['L_thigh'] = compute_segment_ag(jc['L_hip'], jc['L_knee'], 'L_med_knee', 'L_lat_knee', 'thigh', joints, neutral)
    ag_neutral['R_thigh'] = compute_segment_ag(jc['R_hip'], jc['R_knee'], 'R_med_knee', 'R_lat_knee', 'thigh', joints, neutral)
    ag_neutral['L_shank'] = compute_segment_ag(jc['L_knee'], jc['L_ankle'], 'L_med_ankle', 'L_lat_ankle', 'shank', joints, neutral)
    ag_neutral['R_shank'] = compute_segment_ag(jc['R_knee'], jc['R_ankle'], 'R_med_ankle', 'R_lat_ankle', 'shank', joints, neutral)

    # Foot has a distinct AG calculation in the original notebook due to marker placement
    if all(m in neutral for m in tracking_marker_sets['L_foot'][:-1]): # Use _1 to _4 for foot AG
        L_foot_markers_n_ag = np.array([neutral[m] for m in tracking_marker_sets['L_foot'][:-1]])
        L_foot_markers_n_ag = L_foot_markers_n_ag[np.argsort(L_foot_markers_n_ag[:, 0])]
        l_foot_x_ref = np.array([0, 0, -1]) # Consistent with notebook
        l_foot_temp_vec = safe_normalize(L_foot_markers_n_ag[1] - L_foot_markers_n_ag[2])
        if l_foot_temp_vec[1] < 0: l_foot_temp_vec = -l_foot_temp_vec
        l_foot_z = safe_normalize(np.cross(l_foot_x_ref, l_foot_temp_vec))
        l_foot_y = safe_normalize(np.cross(l_foot_z, l_foot_x_ref))
        l_foot_x = safe_normalize(np.cross(l_foot_y, l_foot_z))
        ag_neutral['L_foot'] = np.column_stack((l_foot_x, l_foot_y, l_foot_z))
    else:
        ag_neutral['L_foot'] = np.full((3,3), np.nan)
        print(f"Warning: Missing neutral markers for Left Foot AG calculation in {filepath}")

    if all(m in neutral for m in tracking_marker_sets['R_foot'][:-1]):
        R_foot_markers_n_ag = np.array([neutral[m] for m in tracking_marker_sets['R_foot'][:-1]])
        R_foot_markers_n_ag = R_foot_markers_n_ag[np.argsort(R_foot_markers_n_ag[:, 0])]
        r_foot_x_ref = np.array([0, 0, -1])
        r_foot_temp_vec = safe_normalize(R_foot_markers_n_ag[0] - R_foot_markers_n_ag[1])
        if r_foot_temp_vec[1] < 0: r_foot_temp_vec = -r_foot_temp_vec
        r_foot_z = safe_normalize(np.cross(r_foot_x_ref, r_foot_temp_vec))
        r_foot_y = safe_normalize(np.cross(r_foot_z, r_foot_x_ref))
        r_foot_x = safe_normalize(np.cross(r_foot_y, r_foot_z))
        ag_neutral['R_foot'] = np.column_stack((r_foot_x, r_foot_y, r_foot_z))
    else:
        ag_neutral['R_foot'] = np.full((3,3), np.nan)
        print(f"Warning: Missing neutral markers for Right Foot AG calculation in {filepath}")


    # --- Step 6: Calculate Neutral Cluster Transformations ---
    # This is R_global_to_cluster_local_neutral. It defines the initial orientation of the cluster's local frame relative to global.
    # We need this to define the R_anatomical_to_cluster_local_neutral = ag_neutral.T @ R_global_to_cluster_local_neutral
    neutral_cluster_transformations = {}
    for segment_name, markers_in_set in tracking_marker_sets.items():
        neutral_cluster_points = np.array([neutral.get(m, np.full(3, np.nan)) for m in markers_in_set])
        if np.any(np.isnan(neutral_cluster_points)) or neutral_cluster_points.shape[0] < 3:
            neutral_cluster_transformations[segment_name] = np.full((3,3), np.nan)
            print(f"Warning: Not enough valid neutral markers for {segment_name} cluster transformation.")
            continue

        # Define a local coordinate system for the neutral cluster points
        # A simple way: centroid as origin, and align to canonical axes
        centroid = np.mean(neutral_cluster_points, axis=0)
        centered_points = neutral_cluster_points - centroid

        # Align to identity for its own local frame. This gives R from its local frame to global.
        # This gives R_cluster_local_to_global_neutral
        try:
            rotation, _ = R_scipy.align_vectors(np.identity(3), centered_points[:3]) # Use first 3 markers for orientation if possible
            neutral_cluster_transformations[segment_name] = rotation.as_matrix().T # R_global_to_cluster_local
        except ValueError: # Occurs if points are collinear or not enough
            neutral_cluster_transformations[segment_name] = np.full((3,3), np.nan)
            print(f"Warning: Could not align neutral markers for {segment_name} cluster transformation (e.g., collinear markers).")


    # Calculate R_anatomical_to_cluster_local_neutral for each segment
    R_anatomical_to_cluster_local_neutral = {}
    for seg_name in ag_neutral.keys():
        if not np.any(np.isnan(ag_neutral[seg_name])) and not np.any(np.isnan(neutral_cluster_transformations.get(seg_name, np.full((3,3),np.nan)))):
            # ag_neutral is R_seg_ACS_to_Global_neutral
            # neutral_cluster_transformations[seg_name] is R_global_to_cluster_local_neutral
            # We want R_seg_ACS_to_cluster_local_neutral
            R_anatomical_to_cluster_local_neutral[seg_name] = neutral_cluster_transformations[seg_name] @ ag_neutral[seg_name]
        else:
            R_anatomical_to_cluster_local_neutral[seg_name] = np.full((3,3), np.nan)


    # --- Step 7: Dynamic Kinematics - Calculate Global Segment Rotations per frame ---
    # Store global segment rotation matrices for each frame
    segment_global_rotations_per_frame = {seg: [] for seg in tracking_marker_sets.keys()}

    for frame_idx in range(num_frames):
        for segment_name, markers_in_set in tracking_marker_sets.items():
            current_dynamic_points = np.array([dynamic_data_raw.get(m, np.full(3, np.nan))[frame_idx] for m in markers_in_set])

            if np.any(np.isnan(current_dynamic_points)) or current_dynamic_points.shape[0] < 3:
                segment_global_rotations_per_frame[segment_name].append(np.full((3,3), np.nan))
                continue

            # Get the neutral cluster points relative to their centroid
            neutral_cluster_points_centered = np.array([neutral.get(m, np.full(3, np.nan)) for m in markers_in_set])
            if np.any(np.isnan(neutral_cluster_points_centered)) or neutral_cluster_points_centered.shape[0] < 3:
                segment_global_rotations_per_frame[segment_name].append(np.full((3,3), np.nan))
                continue

            centroid_neutral_cluster = np.mean(neutral_cluster_points_centered, axis=0)
            centered_neutral_cluster_points = neutral_cluster_points_centered - centroid_neutral_cluster

            # Get the dynamic cluster points relative to their centroid
            centroid_dynamic_cluster = np.mean(current_dynamic_points, axis=0)
            centered_dynamic_points = current_dynamic_points - centroid_dynamic_cluster

            # Find the rotation from neutral cluster to current dynamic cluster (in global space)
            # R_dynamic_cluster_to_neutral_cluster
            try:
                R_cluster_dynamic_to_neutral, _ = R_scipy.align_vectors(centered_dynamic_points, centered_neutral_cluster_points)
                # This gives R_dynamic_cluster_to_neutral_cluster, we need R_global_to_dynamic_cluster_local
                # The rotation of the current cluster frame relative to the neutral cluster frame
                R_cluster_relative_to_neutral = R_cluster_dynamic_to_neutral.inv() # <-- CORRECTED LINE
                # R_neutral_cluster_to_dynamic_cluster

                # Combine with anatomical-to-cluster transformation and global neutral cluster orientation
                # R_segment_ACS_to_Global_Current = R_cluster_current_to_Global @ R_TCS_to_ACS_neutral
                # R_segment_ACS_to_Global_Current = R_cluster_current_to_Global @ np.linalg.inv(R_ACS_to_TCS_neutral)
                R_cluster_current_to_Global, _ = R_scipy.align_vectors(centered_neutral_cluster_points, centered_dynamic_points)

                if not np.any(np.isnan(ag_neutral.get(segment_name, np.full((3,3),np.nan)))) and \
                   not np.any(np.isnan(neutral_cluster_transformations.get(segment_name, np.full((3,3),np.nan)))):

                    R_ACS_to_TCS_neutral = np.linalg.inv(neutral_cluster_transformations[segment_name]) @ ag_neutral[segment_name]

                    R_segment_ACS_to_Global_current = R_cluster_current_to_Global.as_matrix() @ R_ACS_to_TCS_neutral

                    segment_global_rotations_per_frame[segment_name].append(R_segment_ACS_to_Global_current)
                else:
                    segment_global_rotations_per_frame[segment_name].append(np.full((3,3), np.nan))

            except ValueError: # align_vectors can fail if points are collinear
                segment_global_rotations_per_frame[segment_name].append(np.full((3,3), np.nan))
                # print(f"Warning: Could not align markers for {segment_name} in frame {frame_idx} (collinear or insufficient markers).")


    # --- Step 8: Calculate Joint Angles per frame ---
    all_joint_angles = []

    # Updated joint_definitions to include Pelvis, Hip, Knee, and Ankle
    joint_definitions = {
        'L_ankle': ('L_shank', 'L_foot'),
        'R_ankle': ('R_shank', 'R_foot'),
        'L_knee': ('L_thigh', 'L_shank'),
        'R_knee': ('R_thigh', 'R_shank'),
        'L_hip': ('pelvis', 'L_thigh'), # Re-added hip
        'R_hip': ('pelvis', 'R_thigh'), # Re-added hip
    }

    for frame_idx in range(num_frames):
        current_frame_angles = {'Frame': frame_idx}

        # Add pelvis global angles if its AG matrix was defined and relevant
        R_pelvis_global = segment_global_rotations_per_frame['pelvis'][frame_idx]
        if not np.any(np.isnan(R_pelvis_global)):
            # Pelvis angles are often reported relative to the global coordinate system
            pelvis_angles_rad = cardanangles(R_pelvis_global)
            pelvis_angles_deg = np.degrees(pelvis_angles_rad)
            current_frame_angles[f'pelvis_X'] = pelvis_angles_deg[0]
            current_frame_angles[f'pelvis_Y'] = pelvis_angles_deg[1]
            # current_frame_angles[f'pelvis_Z'] = pelvis_angles_deg[2] # Z-angle commented out for pelvis as well
        else:
            current_frame_angles[f'pelvis_X'] = np.nan
            current_frame_angles[f'pelvis_Y'] = np.nan
            # current_frame_angles[f'pelvis_Z'] = np.nan # Z-angle commented out for pelvis as well


        for joint_name, (proximal_seg, distal_seg) in joint_definitions.items():
            R_prox_global = segment_global_rotations_per_frame[proximal_seg][frame_idx]
            R_dist_global = segment_global_rotations_per_frame[distal_seg][frame_idx]

            if not np.any(np.isnan(R_prox_global)) and not np.any(np.isnan(R_dist_global)):
                # R_relative = R_prox_global.T @ R_dist_global (standard convention: distal w.r.t proximal)
                R_relative = np.linalg.inv(R_prox_global) @ R_dist_global
                angles_rad = cardanangles(R_relative)
                angles_deg = np.degrees(angles_rad)
            else:
                angles_deg = np.full(3, np.nan) # Fill with NaNs if segment rotations are not available

            current_frame_angles[f'{joint_name}_X'] = angles_deg[0]
            current_frame_angles[f'{joint_name}_Y'] = angles_deg[1]
            # current_frame_angles[f'{joint_name}_Z'] = angles_deg[2] # Z-angle (axial rotation) commented out as requested

        all_joint_angles.append(current_frame_angles)

    angles_df = pd.DataFrame(all_joint_angles)

    if angles_df.empty:
        print(f"No valid angle data generated for {filepath}.")

    return angles_df

# # --- Example of how to use the function to loop through files ---
# if __name__ == '__main__':
#       import os

#       # IMPORTANT: Replace with the actual path to your folder containing JSON data files
#       data_directory = 'path/to/your/json/data_folder/' # e.g., 'C:/Users/YourUser/Documents/GaitData/'

#       # Optional: Directory to save processed DataFrames
#       output_df_directory = 'processed_angle_dfs/'
#       os.makedirs(output_df_directory, exist_ok=True)

#       # Get a list of all JSON files in the directory
#       json_files = [f for f in os.listdir(data_directory) if f.lower().endswith('.json')]

#       if not json_files:
#           print(f"No JSON files found in '{data_directory}'. Please check the path and file types.")
#       else:
#           print(f"Found {len(json_files)} JSON files to process.")

#       all_processed_angles = {} # To store the angle DataFrames from all files

#       for json_file in json_files:
#           full_file_path = os.path.join(data_directory, json_file)

#           print(f"\nProcessing: {json_file}")

#           # Call the function to process the JSON data and get the angles DataFrame
#           angles_df = analyze_gait_data(full_file_path)

#           if not angles_df.empty: # Check if angles_df is not empty
#               file_base_name = os.path.splitext(json_file)[0]
#               all_processed_angles[file_base_name] = angles_df
#               print(f"Successfully processed {json_file}. Generated DataFrame with {len(angles_df)} frames.")

#               # Save the angles DataFrame to a CSV file for each processed file
#               output_csv_path = os.path.join(output_df_directory, f'{file_base_name}_joint_angles_pelvis_hip_knee_ankle_no_z.csv')
#               angles_df.to_csv(output_csv_path, index=False)
#               print(f"Saved joint angles to {output_csv_path}")
#           else:
#               print(f"Failed to process or no valid angle data for {json_file}. No CSV saved.")

#       print("\nAll JSON files processed.")