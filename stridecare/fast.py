#streamlit calls to here, we preprocess things here using functions defined in preprocessing.py
#fast.api
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

#from stridecare.preprocessor import preprocess_features
#from stridecare.mediapipe_cardan_angles.final_mediapipe_angles import get_mediapipe_angles


app = FastAPI()

# @app.get("/predict")
# def predict(
#         pickup_datetime: str,  # 2014-07-06 19:18:00
#         pickup_longitude: float,    # -73.950655
#         pickup_latitude: float,     # 40.783282
#         dropoff_longitude: float,   # -73.984365
#         dropoff_latitude: float,    # 40.769802
#         passenger_count: int
#     ):      # 1
#     """
#     Make a single course prediction.
#     Assumes `pickup_datetime` is provided as a string by the user in "%Y-%m-%d %H:%M:%S" format
#     Assumes `pickup_datetime` implicitly refers to the "US/Eastern" timezone (as any user in New York City would naturally write)
#     """
#     X_pred = pd.DataFrame(dict(
#         pickup_datetime=[pd.Timestamp(pickup_datetime, tz='US/Eastern')],
#         pickup_longitude=[pickup_longitude],
#         pickup_latitude=[pickup_latitude],
#         dropoff_longitude=[dropoff_longitude],
#         dropoff_latitude=[dropoff_latitude],
#         passenger_count=[passenger_count],
#     ))

#     model = load_model()
#     assert model is not None

#     X_processed = preprocess_features(X_pred)
#     y_pred = model.predict(X_processed)

#     print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")

#     return {'fare' : float(y_pred[0][0])}

# @app.get("/")
# def root():
#     return {'greeting': 'Hello'}
