import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image

app = FastAPI()
model = tf.keras.models.load_model("cat_dog_model.keras")

THRESHOLD = 0.85  # tune this: higher = more "unknown", lower = more forced guesses

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
    img = Image.open(io.BytesIO(await file.read())).convert("RGB").resize((128, 128))
    x = np.array(img, dtype="float32")[None, ...]   # raw 0-255; Rescaling is inside the model
    p = float(model.predict(x, verbose=0)[0][0])

    label = "dog" if p > 0.5 else "cat"
    confidence = p if p > 0.5 else 1 - p

    if confidence < THRESHOLD:
        label = "unknown"

    return {"label": label, "confidence": confidence}
