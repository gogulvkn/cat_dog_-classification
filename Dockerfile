FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=8000

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake ImageNet gate weights into the image (no download at startup)
RUN python -c "from tensorflow import keras; keras.applications.MobileNetV2(weights='imagenet', include_top=True)"

COPY app ./app
COPY models ./models

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
