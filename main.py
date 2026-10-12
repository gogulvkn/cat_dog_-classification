import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

app = FastAPI()
model = tf.keras.models.load_model("cat_dog_model.keras")
gate = MobileNetV2(weights="imagenet")  # downloads once (~14 MB)

# ImageNet class ranges: 151-268 = dog breeds, 281-285 = domestic cats
CAT_DOG_IDX = list(range(151, 269)) + list(range(281, 286))
GATE_MIN = 0.5  # min probability mass on cat/dog classes to accept

PAGE = """
<html><body style="font-family:sans-serif;max-width:480px;margin:40px auto">
<h2>Cat vs Dog Classifier</h2>
<input type="file" id="f" accept="image/*"><br><br>
<img id="p" style="max-width:100%"><p id="r"></p>
<script>
f.onchange = async () => {
  const file = f.files[0]; p.src = URL.createObjectURL(file); r.innerText = "Predicting...";
  const fd = new FormData(); fd.append("file", file);
  const res = await (await fetch("/predict", {method:"POST", body:fd})).json();
  r.innerText = res.label + " (" + (res.confidence*100).toFixed(1) + "%)";
};
</script></body></html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")

    # Step 1: gatekeeper - is this a cat or dog at all?
    g = np.array(img.resize((224, 224)), dtype="float32")[None, ...]
    gp = gate.predict(preprocess_input(g), verbose=0)[0]
    catdog_score = float(gp[CAT_DOG_IDX].sum())
    if catdog_score < GATE_MIN:
        return {"label": "unknown", "confidence": 1 - catdog_score}

    # Step 2: your cat vs dog model
    x = np.array(img.resize((128, 128)), dtype="float32")[None, ...]
    p = float(model.predict(x, verbose=0)[0][0])
    return {"label": "dog" if p > 0.5 else "cat",
            "confidence": p if p > 0.5 else 1 - p}
