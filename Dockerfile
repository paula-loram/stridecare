FROM tensorflow/tensorflow:2.16.1
COPY api /api
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
CMD uvicorn api.main:app --host 0.0.0.0 --port $PORT
