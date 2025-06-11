FROM tensorflow/tensorflow:2.16.1
RUN apt-get update && apt-get install -y ffmpeg libavcodec-extra
COPY api /api
COPY requirements_prod.txt /requirements.txt
RUN pip install -r requirements.txt
CMD uvicorn api.main:app --host 0.0.0.0 --port $PORT

