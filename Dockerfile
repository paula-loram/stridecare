FROM python:3.10.6-slim
COPY stridecare /stridecare
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
CMD uvicorn stridecare.api.main:app --host 0.0.0.0 --port $PORT
