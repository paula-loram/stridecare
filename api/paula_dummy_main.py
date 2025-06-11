import os
import tempfile
import json
import numpy as np
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Literal
from starlette.responses import JSONResponse
import tensorflow as tf

import tempfile
from google.cloud import storage

# --- Import your custom modules ---
from video_angle_processor import get_mediapipe_angles
from api.preprocessing import load_scalers, preprocess_angles, preprocess_metadata
from api.preprocessing import cat_metadata_scaler, numerical_metadata_scaler  # to check scaler status in health check
from load_model import load_model

# --- Configuration & Model Path ---
MODEL_PATH = "./RNN/my_model_weights.weights.h5" #----------> get bucket address
TEMP_VIDEO_DIR = "temp_videos"

# Global variable for the loaded model
model = None

# Model output labels and their order
MODEL_OUTPUT_LABELS = [
    "No injury",
    "Knee",
    "Foot/Ankle",
    "Hip/Pelvis",
    "Thigh",
    "Lower Leg"
]

# Pydantic model for incoming metadata validation
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

app.state.model = load_model() #---------> new
app.state.scalers = load_scalers() #----------> new

@app.get("/")
def root():
    return {'status': 'backend up!'}

# @app.get("/health")
# async def health_check():
#     status = "healthy"
#     detail = "All assets loaded."

#     if model is None:
#         status = "degraded"
#         detail = "Model not loaded."
#     elif cat_metadata_scaler is None or numerical_metadata_scaler is None:
#         status = "degraded"
#         detail = "Scalers not loaded."

#     return {
#         "status": status,
#         "model_loaded": model is not None,
#         "cat_metadata_scaler_loaded": cat_metadata_scaler is not None,
#         "numerical_metadata_scaler_loaded": numerical_metadata_scaler is not None,
#         "detail": detail
#     }

@app.post("/get_stick_fig_video")
async def upload_video(background_tasks: BackgroundTasks, video: UploadFile = File(...)):
    if not video:
        return JSONResponse(status_code=400, content={"message": "No video file uploaded."})
    if not video.filename or not video.content_type:
        return JSONResponse(status_code=400, content={"message": "Filename or content type missing."})
    if not video.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.flv')):
        return JSONResponse(status_code=400, content={"message": "Unsupported video format."})

    os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)

    temp_file_path = os.path.join(TEMP_VIDEO_DIR, video.filename)

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        return JSONResponse(status_code=200, content={
            "message": "Video received and processing started in the background.",
            "filename": video.filename,
            "content_type": video.content_type,
            "temp_path": temp_file_path
        })
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return JSONResponse(status_code=500, content={"message": f"Failed to save video: {e}"})

@app.post("/predict") #-----------> we don't need to load video anymore
async def predict_injury_risk(
    video: UploadFile = File(...),
    metadata: str = Form(...)
):

    model = app.state.model #------> load model
    scalers = app.state.scalers #------> load scalers from above
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Server not ready.")

    # Parse metadata JSON
    try:
        metadata_dict = json.loads(metadata)
        user_meta = UserMetadata(**metadata_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid metadata provided: {e}")

    # Save video to temporary file
    temp_video_path = None
    try:
        suffix_map = {
            "video/mp4": ".mp4",
            "video/quicktime": ".mov",
            "video/x-msvideo": ".avi",
        }
        file_suffix = suffix_map.get(video.content_type, ".mp4")

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_file:
            temp_file.write(await video.read())
            temp_video_path = temp_file.name

        # Extract angles
        #angles_array = get_mediapipe_angles(temp_video_path) #--------> we dont need to use this anymore, we need the angles from

        # Preprocess angles and metadata
        processed_angles = preprocess_angles(angles_array)
        processed_metadata = preprocess_metadata(user_meta.age, user_meta.weight, user_meta.height, user_meta.gender)

        if processed_angles.size == 0 or np.any(np.isnan(processed_angles)) or \
           processed_metadata.size == 0 or np.any(np.isnan(processed_metadata)):
            raise HTTPException(status_code=422, detail="Failed to preprocess video or metadata. Data may be incomplete or invalid after scaling/padding.")

        # Make prediction
        prediction_raw = model.predict([processed_angles, processed_metadata])

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
            "message": "Video analyzed and prediction made.",
            "prediction": predicted_label,
            "confidence": confidence,
            "all_class_probabilities": all_class_probabilities,
            "details": {
                "angles_input_shape": processed_angles.shape,
                "metadata_input_shape": processed_metadata.shape,
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
