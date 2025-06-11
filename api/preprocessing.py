#should have the preprocessing of input data -- vid to angle
#pickled scaler and ohe
#and here is where we calculate the prediction with the model
#import video_angle_preprocessor to have get_mediapipe_angles() available

import numpy as np
import pandas as pd
import joblib # To load pre-trained scikit-learn scalers
import os
from api.video_angle_processor import get_mediapipe_angles
from google.cloud import storage


# --- Configuration ---
# Path to saved scalers. These paths are relative to the 'stridecare/' root
# when the Docker container's WORKDIR is set to '/app'.

client = storage.Client()
bucket_name = 'stridecare-models'
bucket = client.get_bucket(bucket_name)
blob_cat = bucket.blob('scalers/ohe.pkl') if bucket else None
blob_num = bucket.blob('scalers/scaler.pkl') if bucket else None

GCS_SCALERS_PREFIX = 'scalers/' #delete?
SCALERS_DIR = "gs://stridecare-models/scalers/" #GCS file path
CAT_METADATA_SCALER_FILENAME = "ohe.pkl" # For OneHotEncoder for gender
NUML_METADATA_SCALER_FILENAME = "scaler.pkl" # For StandardScaler for age, weight, height

# Fixed frame length for the RNN
RNN_SEQUENCE_LENGTH = 6000 # PADDING FOR FRAMES

# --- Global variables for loaded scalers ---
cat_metadata_scaler = None
numerical_metadata_scaler = None

def load_scalers():
    """
    Loads pre-trained scikit-learn scalers from disk.
    This function should be called once at application startup.
    """
    global cat_metadata_scaler, numerical_metadata_scaler
    try:
        cat_metadata_scaler_path = os.path.join(SCALERS_DIR, CAT_METADATA_SCALER_FILENAME)
        numerical_metadata_scaler_path = os.path.join(SCALERS_DIR, NUML_METADATA_SCALER_FILENAME)

        if not os.path.exists(cat_metadata_scaler_path):
            raise FileNotFoundError(f"OneHotEncoder scaler not found at: {cat_metadata_scaler_path}")
        if not os.path.exists(numerical_metadata_scaler_path):
            raise FileNotFoundError(f"Numerical metadata scaler not found at: {numerical_metadata_scaler_path}")

        cat_metadata_scaler = joblib.load(cat_metadata_scaler_path)
        numerical_metadata_scaler = joblib.load(numerical_metadata_scaler_path)
        print("Preprocessing: Successfully loaded OneHotEncoder and numerical metadata scalers.")
        return True

    except Exception as e:
        print(f"Preprocessing Error: Could not load scalers. Ensure paths are correct and files exist. Error: {e}")
        cat_metadata_scaler = None
        numerical_metadata_scaler = None
        return False

def preprocess_metadata(age: int, height: float, weight: float, gender: str) -> np.array:
    """
    Applies the same preprocessing (rounding, scaling, encoding, concatenation)
    to metadata as during training.

    Args:
        age (int): User's age.
        weight (float): User's weight in kg.
        height (float): User's height in cm.
        gender (str): User's gender ("Male", "Female", "Other").

    Returns:
        np.array: Preprocessed metadata ready for the RNN model,
                  reshaped to (1, num_meta_features).
                  Returns an array of NaNs if scalers are not loaded.
    """
    if numerical_metadata_scaler is None or cat_metadata_scaler is None:
        print("Preprocessing: Metadata scalers not loaded. Cannot preprocess metadata.")
        # Return a NaN-filled array with expected number of metadata features
        # (3 numerical + number of gender categories from OHE)
        num_expected_meta_features = 3 + len(cat_metadata_scaler.categories_[0]) if cat_metadata_scaler else 6
        return np.full((1, num_expected_meta_features), np.nan)

    # 1. Round numerical values to one decimal point
    age_rounded = round(float(age), 1) # Ensure float conversion before rounding
    weight_rounded = round(float(weight), 1)
    height_rounded = round(float(height), 1)

    # Create a DataFrame for numerical features, maintaining column order
    numerical_df = pd.DataFrame([{
        'age': age_rounded,
        'height': height_rounded,
        'weight': weight_rounded
    }])

    # Ensure column order for numerical features matches what numerical_metadata_scaler expects
    # This list MUST match the columns numerical_metadata_scaler was fitted on, in order.
    numerical_cols_order = ['age', 'height', 'weight']
    numerical_data_to_scale = numerical_df[numerical_cols_order].values

    # 2. Standard scale numerical features
    scaled_numerical = numerical_metadata_scaler.transform(numerical_data_to_scale)

    # 3. One-hot encode gender
    # The OneHotEncoder expects a 2D array of categories.
    gender_input_for_ohe = np.array([[gender]])

    try:
        # .toarray() converts sparse matrix output to dense numpy array
        encoded_gender = cat_metadata_scaler.transform(gender_input_for_ohe).toarray()
    except ValueError as e:
        # Handle unseen gender categories during inference.
        # If a gender is encountered that wasn't in training, OHE will raise an error.
        # Common approach: treat as 'Other' or create an all-zero vector for that category.
        print(f"Preprocessing Warning: Gender category '{gender}' not seen during OneHotEncoder training. Error: {e}")
        # Create an all-zero vector for the one-hot encoded gender if category is unseen.
        # This assumes the OHE has at least one category, so categories_[0] is safe.
        num_gender_categories = len(cat_metadata_scaler.categories_[0])
        encoded_gender = np.zeros((1, num_gender_categories))

    # 4. Concatenate scaled numerical features and one-hot encoded gender
    # The order of concatenation MUST match the order expected by your model
    # (e.g., numerical features first, then one-hot encoded gender features)
    processed_meta = np.concatenate((scaled_numerical, encoded_gender), axis=1)

    # Reshape for model input: (batch_size, num_meta_features)
    # For a single inference, batch_size is 1, so the shape is already (1, num_features).
    return processed_meta

def preprocess_angles(raw_angles_array: np.array) -> np.array:
    """
    Applies the specified preprocessing (division by 180, imputation, padding/truncation)
    to angles data.

    Args:
        raw_angles_array (np.array): Raw angles data (num_frames, num_features)
                                     from get_mediapipe_angles.

    Returns:
        np.array: Preprocessed angles ready for the RNN model,
                  reshaped to (1, RNN_SEQUENCE_LENGTH, num_features).
                  Returns an array of NaNs if input is invalid or expected shape mismatch.
    """
    num_expected_features = 10 # 2pelvis, 4 hips, 4 knees

    # If no angle data was extracted for the video, create a NaN array for processing
    if raw_angles_array.size == 0:
        print("Preprocessing: No raw angle data found; creating a NaN-filled sequence.")
        temp_angles = np.full((RNN_SEQUENCE_LENGTH, num_expected_features), np.nan)
    else:
        # Ensure raw_angles_array has the expected number of columns for robust operation
        if raw_angles_array.shape[1] != num_expected_features:
            print(f"Warning: Raw angles array has {raw_angles_array.shape[1]} features, expected {num_expected_features}. Adjusting or padding.")
            # Create a new array and fill with NaNs, then copy valid columns
            adjusted_angles = np.full((raw_angles_array.shape[0], num_expected_features), np.nan)
            min_cols = min(raw_angles_array.shape[1], num_expected_features)
            adjusted_angles[:, :min_cols] = raw_angles_array[:, :min_cols]
            temp_angles = adjusted_angles
        else:
            temp_angles = raw_angles_array.copy() # Work on a copy to avoid modifying original

    # 1. Standardize by dividing by 180 (each value)
    # Handle division by zero for NaN or zero values by ensuring 180.0 is float
    standardized_angles = temp_angles / 180.0

    #Round to 8 decimal points to match training
    rounded_angles = np.round(standardized_angles, 8)

    # 2. Imputation: Fill any remaining NaNs
    processed_angles = np.nan_to_num(rounded_angles, nan=0.0) # Using 0.0 as a common default

    # 3. Padding/Truncation: Ensure sequence length matches RNN_SEQUENCE_LENGTH
    if processed_angles.shape[0] > RNN_SEQUENCE_LENGTH:
        # Truncate from the end (or start, or middle, depending on your training strategy)
        processed_angles = processed_angles[:RNN_SEQUENCE_LENGTH, :]
    else:
        # Pad with zeros (or a specific padding value if used during training).
        # Match your training data padding strategy (pre-padding or post-padding)!
        padding_needed = RNN_SEQUENCE_LENGTH - processed_angles.shape[0]

        processed_angles = np.pad(processed_angles,
                                  ((0, padding_needed), (0, 0)),
                                  mode='constant', constant_values=0.0)

    return processed_angles[np.newaxis, :, :]
