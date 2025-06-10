import tempfile
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Literal
from starlette.responses import JSONResponse
import tensorflow as tf
from get_stickfigure import get_stickfigure
import base64

app = FastAPI(
    title="Stridecare Running Project API",
    description="API for video-based injury risk prediction using MediaPipe and RNN.",
    version="1.0.0")

@app.get("/")
def root():
    return {
        'status' : 'backend up!'
    }

# --- API Endpoint for Stick Figure Generation ---
@app.post("/generate_stickfigure")
async def generate_stickfigure_api(video: UploadFile = File(...)):
    """
    Receives a video file, generates a stick figure representation with MediaPipe Pose,
    and returns the base64 encoded stick figure video.
    """
    temp_video_path = None
    stick_figure_output_path = None
    encoded_stick_figure_video = ""

    # Save the uploaded video to a temporary file
    suffix_map = { "video/mp4": ".mp4", "video/quicktime": ".mov", "video/x-msvideo": ".avi", }
    file_suffix = suffix_map.get(video.content_type, ".mp4")

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_file:
        temp_file.write(await video.read())
        temp_video_path = temp_file.name
    print(f"Backend (/generate_stickfigure): Original video saved temporarily to: {temp_video_path}")

    # Define output path for the stick figure video
    stick_figure_output_path = temp_video_path + "_stickfigure.mp4"

    # Call the get_stickfigure_video function
    generated_path = get_stickfigure(
        video_path=temp_video_path,
        output_path=stick_figure_output_path
    )

    if generated_path and os.path.exists(generated_path):
        print(f"Backend (/generate_stickfigure): Stick figure video generated to: {generated_path}")
        # Read the generated video bytes and base64 encode them
        with open(generated_path, "rb") as video_file_bytes:
            encoded_stick_figure_video = base64.b64encode(video_file_bytes.read()).decode('utf-8')
    else:
        print("Backend (/generate_stickfigure) Warning: Failed to generate stick figure video.")
        raise HTTPException(status_code=500, detail="Failed to generate stick figure video.")

    return JSONResponse(content={
        "message": "Stick figure video generated successfully.",
        "stick_figure_video_b64": encoded_stick_figure_video, # Send the base64 encoded video
    })


@app.post("/predict")
async def predict_injury(video: UploadFile = File(...),
                         metadata: str = Form(...)): #sent as JSON

    
