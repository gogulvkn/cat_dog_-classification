# Cat vs Dog Image Classifier

A deep learning project that classifies images as **cat** or **dog** using a convolutional neural network (CNN) built with TensorFlow/Keras, served through a FastAPI web app.

## Features
- Binary image classification (cat / dog) with a confidence score
- Data augmentation and dropout to reduce overfitting
- Optional transfer learning with MobileNetV2
- Simple web interface for uploading an image and getting a prediction
- REST endpoint: `POST /predict`

## Tech Stack
Python, TensorFlow/Keras, FastAPI, Uvicorn, Pillow, NumPy, Render

## Dataset
[Cat vs Dog Image Classification](https://www.kaggle.com/datasets/vishallazrus/cat-vs-dog-image-classification-making-prediction) from Kaggle.

## Model
- Input: 128x128 RGB images
- Layers: Conv2D + MaxPooling blocks, Flatten, Dense(128), Dropout, sigmoid output
- Loss: binary cross-entropy, Optimizer: Adam
- Validation accuracy: XX%  <!-- fill in from model.evaluate -->

## Run Locally
```bash
git clone https://github.com/<your-username>/catdog-app.git
cd catdog-app
pip install -r requirements.txt
uvicorn main:app --reload
```
Open http://127.0.0.1:8000 and upload an image.

## API
```bash
curl -X POST -F "file=@cat.jpg" http://127.0.0.1:8000/predict
# {"label": "cat", "confidence": 0.98}
```

## Project Structure
