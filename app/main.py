"""FastAPI service for the cat/dog classifier.

Model facts (from the training notebook):
  * MobileNetV2 transfer model, input 160x160x3, raw 0-255 pixels
    (Rescaling + augmentation layers are INSIDE the model, so do NOT rescale here)
  * Single sigmoid output: 0 = cat, 1 = dog

The binary model can never say "unknown" on its own, so two guards are applied:
  1. ImageNet gate  - a pretrained MobileNetV2 checks the image is actually a cat/dog
                      (fixes horse -> "dog"). Disable with USE_IMAGENET_GATE=0.
  2. Confidence threshold on the classifier output.
If you later retrain a 3-class softmax model (cat, dog, unknown), it is detected
automatically from the output shape.
"""
import os
from contextlib import asynccontextmanager

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from tensorflow import keras

MODEL_PATH = os.getenv("MODEL_PATH", "models/cat_dog_model.keras")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.80"))
USE_GATE = os.getenv("USE_IMAGENET_GATE", "1") == "1"
GATE_THRESHOLD = float(os.getenv("GATE_THRESHOLD", "0.50"))
MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
CLASS_NAMES_3 = os.getenv("CLASS_NAMES", "cat,dog,unknown").split(",")

# ImageNet index ranges: 151-268 = dog breeds, 281-285 = domestic cats
DOG_IDX = np.arange(151, 269)
CAT_IDX = np.arange(281, 286)
ANIMAL_IDX = np.concatenate([DOG_IDX, CAT_IDX])

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    model = keras.models.load_model(MODEL_PATH)
    state["model"] = model
    _, h, w, _ = model.input_shape
    state["size"] = (h, w)
    state["units"] = model.output_shape[-1]
    if USE_GATE:
        state["gate"] = keras.applications.MobileNetV2(weights="imagenet", include_top=True)
    yield
    state.clear()


app = FastAPI(title="Cat / Dog Classifier", version="1.0.0", lifespan=lifespan)


def decode(data: bytes) -> np.ndarray:
    try:
        img = tf.io.decode_image(data, channels=3, expand_animations=False)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or unsupported image file")
    return img.numpy()


def resize(img: np.ndarray, size) -> np.ndarray:
    return tf.image.resize(img, size).numpy()[None, ...].astype("float32")


def classify(img: np.ndarray) -> dict:
    model = state["model"]
    out = model.predict(resize(img, state["size"]), verbose=0)[0]

    if state["units"] == 1:  # sigmoid: 0 = cat, 1 = dog
        p_dog = float(out[0])
        probs = {"cat": 1 - p_dog, "dog": p_dog}
    else:  # softmax (cat, dog, unknown)
        probs = {n: float(p) for n, p in zip(CLASS_NAMES_3, out)}

    label = max(probs, key=probs.get)
    conf = probs[label]
    if conf < CONF_THRESHOLD:
        label = "unknown"
    return {"label": label, "confidence": round(conf, 4),
            "probabilities": {k: round(v, 4) for k, v in probs.items()}}


def gate_score(img: np.ndarray) -> float:
    x = resize(img, (224, 224))
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    p = state["gate"].predict(x, verbose=0)[0]
    return float(p[ANIMAL_IDX].sum())


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Cat / Dog Classifier</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{font-family:system-ui,sans-serif;max-width:480px;margin:40px auto;padding:0 16px;text-align:center}
img{max-width:100%;max-height:320px;margin:16px 0;border-radius:8px}
button{padding:10px 20px;font-size:16px;cursor:pointer}
#out{font-size:22px;margin-top:16px}
</style></head><body>
<h2>Cat / Dog Classifier</h2>
<input type="file" id="f" accept="image/*"><br>
<img id="p" style="display:none"><br>
<button onclick="go()">Predict</button>
<div id="out"></div>
<script>
const f=document.getElementById('f'),p=document.getElementById('p'),o=document.getElementById('out');
f.onchange=()=>{if(f.files[0]){p.src=URL.createObjectURL(f.files[0]);p.style.display='inline';o.textContent='';}};
async function go(){
  if(!f.files[0]){o.textContent='Choose an image first';return;}
  o.textContent='Working...';
  const fd=new FormData();fd.append('file',f.files[0]);
  try{
    const r=await fetch('/predict',{method:'POST',body:fd});
    const j=await r.json();
    o.textContent=r.ok?j.label.toUpperCase()+' ('+(j.confidence*100).toFixed(1)+'%)':(j.detail||'Error');
  }catch(e){o.textContent='Request failed';}
}
</script></body></html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    return PAGE


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    data = file.file.read(MAX_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    img = decode(data)

    if USE_GATE:
        score = gate_score(img)
        if score < GATE_THRESHOLD:
            return {"label": "unknown", "confidence": round(1 - score, 4),
                    "reason": "not recognised as a cat or dog",
                    "cat_dog_score": round(score, 4)}

    result = classify(img)
    if USE_GATE:
        result["cat_dog_score"] = round(score, 4)
    return result