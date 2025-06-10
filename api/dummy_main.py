#streamlit calls to here, we preprocess things here using functions defined in preprocessing.py

import os
import tempfile
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Literal
from starlette.responses import JSONResponse
import tensorflow as tf


app = FastAPI(
    title="Stridecare Running Project API",
    description="API for video-based injury risk prediction using MediaPipe and RNN.",
    version="1.0.0"
)

@app.get("/")
def root():
    return {
        'status' : 'backend up!'
    }

#FROM BUCKETS

#MetaData

##Scaler

##OHE


#Preprocess
#Model
