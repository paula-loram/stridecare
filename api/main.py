#streamlit calls to here, we preprocess things here using functions defined in preprocessing.py

import os
import tempfile
import json
import numpy as np
import pandas as pd
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi import Body
from pydantic import BaseModel
from typing import Literal
from starlette.responses import JSONResponse, StreamingResponse
import tensorflow as tf
import base64

# --- Import your custom modules ---
from api.video_angle_processor import get_mediapipe_angles
from api.preprocessing import load_scalers, preprocess_angles, preprocess_metadata
from api.get_stickfigure import get_stickfigure
from api.load_model import load_model

# --- Configuration & Model Path ---
#will be bucket
MODEL_PATH = "./RNN/my_model_weights.weights.h5"
TEMP_VIDEO_DIR = "temp_videos"
# Global variable for the loaded model
model = None

# --- Model's output labels and their order --- --------> DOES ORDER MATTER?
MODEL_OUTPUT_LABELS = [
    "No injury",
    "Knee",
    "Foot/Ankle",
    "Hip/Pelvis",
    "Thigh",
    "Lower Leg"
]

# --- Pydantic Model for Incoming Metadata ---
class UserMetadata(BaseModel):
    age: int
    weight: float
    height: float
    gender: Literal["Male", "Female", "Other"]

app = FastAPI(
    title="Stridecare Running Project API",
    description="API for video-based injury risk prediction using MediaPipe and RNN.",
    version="1.0.0"
)
app.state.model = load_model()  # Load the model at startup
# app.state.cat_metadata_scaler, app.state.numerical_metadata_scaler = load_scalers()  # Load the scalers at startup

@app.get("/")
def root():
    return {
        'status' : 'backend up!'
    }

@app.post("/get_stick_fig_video")
async def upload_video(background_tasks: BackgroundTasks, video: UploadFile = File(...)):
    """
    Receives a video file, saves it temporarily, and then
    starts a background task to process it with MediaPipe.
    """
    # return file name and content type
    if not video:
        return JSONResponse(status_code=400, content={"message": "No video file uploaded."})
    # Check if the uploaded file is a video
    if not video.filename:
        return JSONResponse(status_code=400, content={"message": "No filename provided."})
    if not video.content_type:
        return JSONResponse(status_code=400, content={"message": "No content type provided."})
    # Ensure the content type is a video format
    if not video.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.flv')):
        return JSONResponse(status_code=400, content={"message": "Unsupported video format. Please upload a valid video file."})

    os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)
    temp_file_path = os.path.join(TEMP_VIDEO_DIR, video.filename)
    output_file_path = os.path.join(TEMP_VIDEO_DIR, f"stickfig_{video.filename}")

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # Process the video to generate stick figure video
        stickfig_path, angles_array = get_stickfigure(temp_file_path, output_file_path)

        # Check if the stick figure video was generated and is not empty
        # if not os.path.exists(stickfig_path) or os.path.getsize(stickfig_path) == 0:
        #     return JSONResponse(status_code=500, content={"message": "Stick figure video was not generated or is empty."})
        with open(stickfig_path, "rb") as f:
            video_bytes = f.read()
            video_b64 = base64.b64encode(video_bytes).decode("utf-8")
        # Convert angles_array to list for JSON serialization
        # angles_list = angles_array.tolist() if hasattr(angles_array, "tolist") else angles_array


        # Replace NaN and inf with 999. for JSON serialization
        angles_clean = np.where(np.isnan(angles_array), 999., angles_array)
        angles_clean = np.where(np.isinf(angles_clean), 999., angles_clean)
        angles_list = angles_clean.tolist()

        return JSONResponse(status_code=200, content={
            "message": "Stick figure video generated successfully.",
            "filename": os.path.basename(stickfig_path),
            "video_base64": video_b64,
            "content_type": "video/mp4",
            "angles_array": angles_list
        })

        # def iterfile(file_path):
        #     with open(file_path, "rb") as file_like:
        #         while True:
        #             chunk = file_like.read(1024 * 1024)
        #             if not chunk:
        #                 break
        #             yield chunk
        # # start a background task to run model prediction using angle_data_list and data TODO
        # print(angles_array)
        # return StreamingResponse(
        #     iterfile(stickfig_path),
        #     media_type="video/mp4",
        #     headers={"Content-Disposition": f"attachment; filename=stickfig_{video.filename}"}
        # )
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return JSONResponse(status_code=500, content={"message": f"Failed to save video: {e}"})

@app.post("/predict")
async def predict_injury_risk(
    data: dict = Body(...)
):
    model = app.state.model
    # cat_metadata_scaler = app.state.cat_metadata_scaler
    # numerical_metadata_scaler = app.state.numerical_metadata_scaler
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Server not ready.")

    try:
        # angles_array = np.array(data["angles"], dtype=np.float32)
        user_meta = UserMetadata(
            age=data["age"],
            weight=data["weight"],
            height=data["height"],
            gender=data["gender"]
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input data: {e}")

    # Preprocess angles and metadata
    processed_angles = preprocess_angles(data["angles"])
    processed_metadata = preprocess_metadata(user_meta.age, user_meta.weight, user_meta.height, user_meta.gender)

    if processed_angles.size == 0 or np.any(np.isnan(processed_angles)) or \
       processed_metadata.size == 0 or np.any(np.isnan(processed_metadata)):
        raise HTTPException(status_code=422, detail="Failed to preprocess angles or metadata. Data may be incomplete or invalid after scaling/padding.")

    # Make prediction
    prediction_raw = model.predict([processed_angles, processed_metadata])
    print(f"Backend: Raw model prediction: {prediction_raw}")
    if prediction_raw.ndim == 2 and prediction_raw.shape[0] == 1 and prediction_raw.shape[1] == len(MODEL_OUTPUT_LABELS):
        predicted_class_index = np.argmax(prediction_raw[0])
        predicted_label = MODEL_OUTPUT_LABELS[predicted_class_index]
        confidence = float(prediction_raw[0][predicted_class_index])
        all_class_probabilities = prediction_raw[0].tolist()
    else:
        predicted_label = "Prediction Error: Unexpected model output shape."
        confidence = 0.0
        all_class_probabilities = []
        print(f"Backend Warning: Unexpected model output shape: {prediction_raw.shape}")

    return JSONResponse(content={
        "message": "Angles analyzed and prediction made.",
        "prediction": predicted_label,
        "confidence": confidence,
        "all_class_probabilities": all_class_probabilities,
        "details": {
            "angles_input_shape": processed_angles.shape,
            "metadata_input_shape": processed_metadata.shape,
        }
    })

# --- FastAPI Startup Event: Load Model and Scalers ---
# @app.on_event("startup")
# async def startup_event():
#     """Load the pre-trained model and scalers when the FastAPI app starts."""
#     global model

#     # 1. Load the model
#     try:
#         model = tf.keras.models.load_model(MODEL_PATH)
#         # Optional: Check model output shape to confirm it matches expectations
#         # print(f"Model output shape: {model.output_shape}")
#         if model.output_shape[-1] != len(MODEL_OUTPUT_LABELS):
#             print(f"WARNING: Model output layer size ({model.output_shape[-1]}) does not match number of defined labels ({len(MODEL_OUTPUT_LABELS)}).")
#         print(f"Server Startup: Successfully loaded model from {MODEL_PATH}")
#     except Exception as e:
#         print(f"Server Startup Error: Could not load model. Ensure path is correct. Error: {e}")
#         model = None # Set to None to indicate failure

#     # 2. Load the scalers using the function from preprocessing.py
#     # This also sets the global scaler variables within preprocessing.py
#     scalers_loaded_successfully = load_scalers()
#     if not scalers_loaded_successfully:
#         print("Server Startup Error: Failed to load preprocessing scalers. API will be degraded.")
#         # In a production environment, you might want to raise an exception here
#         # to prevent the server from starting if essential preprocessing assets can't be loaded.

# # --- API Endpoint ---
# @app.post("/predict")
# async def predict_injury_risk(
#     video: UploadFile = File(...),
#     metadata: str = Form(...) # Metadata sent as a JSON string in a form field
# ):
#     """
#     Receives a video file and user metadata, processes them, and returns an injury risk prediction.
#     """
#     # 1. Validate loaded assets
#     if model is None:
#         raise HTTPException(status_code=500, detail="Model not loaded. Server not ready.")
#     # Preprocessing functions will check their internal scalers.

#     # 2. Parse Metadata
#     try:
#         metadata_dict = json.loads(metadata)
#         user_meta = UserMetadata(**metadata_dict) # Validate with Pydantic model
#     except json.JSONDecodeError:
#         raise HTTPException(status_code=400, detail="Invalid metadata JSON format.")
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Invalid metadata provided: {e}")

#     # 3. Save Video to a Temporary File
#     temp_video_path = None
#     try:
#         suffix_map = {
#             "video/mp4": ".mp4",
#             "video/quicktime": ".mov",
#             "video/x-msvideo": ".avi",
#         }
#         file_suffix = suffix_map.get(video.content_type, ".mp4") # Default to .mp4

#         with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_file:
#             temp_file.write(await video.read()) # Read bytes from UploadFile and write to temp file
#             temp_video_path = temp_file.name

#         print(f"Backend: Video saved temporarily to: {temp_video_path}")

#         # 4. Extract Angles using angle_processor.py
#         angles_array = get_mediapipe_angles(temp_video_path)
#         print(f"Backend: Extracted angles shape: {angles_array.shape if angles_array.size > 0 else 'Empty'}")

#         # 5. Preprocess Data for Model using preprocessing.py
#         processed_angles = preprocess_angles(angles_array)
#         processed_metadata = preprocess_metadata(user_meta.age, user_meta.weight, user_meta.height, user_meta.gender)

#         # Check if preprocessing produced valid data
#         if processed_angles.size == 0 or np.any(np.isnan(processed_angles)) or \
#            processed_metadata.size == 0 or np.any(np.isnan(processed_metadata)):
#              raise HTTPException(status_code=422, detail="Failed to preprocess video or metadata. Data may be incomplete or invalid after scaling/padding.")

#         print(f"Backend: Preprocessed angles shape: {processed_angles.shape}")
#         print(f"Backend: Preprocessed metadata shape: {processed_metadata.shape}")

#         # 6. Make Prediction ---------------> how do we do this?
#         try:
#             # Ensure input shapes match your model's expected inputs
#             # Assuming model.predict takes a list of inputs: [angles_input, metadata_input]
#             # where angles_input is (batch_size, seq_len, features)
#             # and metadata_input is (batch_size, meta_features)
#             prediction_raw = model.predict([processed_angles, processed_metadata])

#             # Assuming model outputs probabilities for each class (e.g., using softmax in last layer)
#             # prediction_raw will likely be a 2D array: [[prob_class0, prob_class1, ..., prob_class5]]
#             if prediction_raw.ndim == 2 and prediction_raw.shape[0] == 1 and prediction_raw.shape[1] == len(MODEL_OUTPUT_LABELS):
#                 predicted_class_index = np.argmax(prediction_raw[0]) # Get index of highest probability
#                 predicted_label = MODEL_OUTPUT_LABELS[predicted_class_index]
#                 # Get the confidence for the predicted label
#                 confidence = float(prediction_raw[0][predicted_class_index])
#                 all_class_probabilities = prediction_raw[0].tolist() # Convert probabilities to a list for JSON

#             else:
#                 # Handle unexpected model output shape
#                 predicted_label = "Prediction Error: Unexpected model output shape."
#                 confidence = 0.0
#                 all_class_probabilities = []
#                 print(f"Backend Warning: Unexpected model output shape: {prediction_raw.shape}")


#             return JSONResponse(content={
#                 "message": "Video analyzed and prediction made successfully.",
#                 "prediction": predicted_label, # The human-readable label
#                 "confidence": confidence,     # Confidence for the predicted label
#                 "all_class_probabilities": all_class_probabilities, # All probabilities for insight
#                 "details": {
#                     "angles_input_shape": processed_angles.shape,
#                     "metadata_input_shape": processed_metadata.shape,
#                     # "raw_model_output": prediction_raw.tolist() # Can include full raw output if needed for debugging
#                 }
#             })

#         except Exception as model_err:
#             print(f"Backend Error: An error occurred during model prediction: {model_err}")
#             raise HTTPException(status_code=500, detail=f"Error during model prediction: {model_err}. Check model inputs and output interpretation.")

#     except Exception as e:
#         print(f"Backend Error: Internal server error during processing: {e}")
#         raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

#     finally:
#         # 7. Clean up the temporary video file
#         if temp_video_path and os.path.exists(temp_video_path):
#             os.unlink(temp_video_path)
#             print(f"Backend: Cleaned up temporary video file: {temp_video_path}")


# # --- Health Check Endpoint ---
# @app.get("/health")
# async def health_check():
#     """Returns a simple health check status, indicating if model and scalers are loaded."""
#     # Note: `angles_scaler` and `metadata_scaler` here are references to global
#     # variables in `preprocessing.py`. Their state is managed by `preprocessing.load_scalers()`.
#     from preprocessing import ohe_scaler, numerical_metadata_scaler # Specific scalers from preprocessing.py
#     status = "healthy"
#     detail = "All assets loaded."

#     if model is None:
#         status = "degraded"
#         detail = "Model not loaded."
#     elif ohe_scaler is None or numerical_metadata_scaler is None: # Check specific scalers
#         status = "degraded"
#         detail = "Scalers not loaded."

#     return {
#         "status": status,
#         "model_loaded": model is not None,
#         "ohe_scaler_loaded": ohe_scaler is not None, # Specific scaler status
#         "numerical_metadata_scaler_loaded": numerical_metadata_scaler is not None, # Specific scaler status
#         "detail": detail
#     }
