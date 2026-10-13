# 🐱🐶 Cat / Dog / Unknown Classifier

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-009688)
![License](https://img.shields.io/badge/license-MIT-green)

A 3-class image classifier (**cat**, **dog**, **unknown**) built with MobileNetV2 transfer learning and served with FastAPI. The `unknown` class lets the model say "this is neither a cat nor a dog" (horse, cow, bird, ...) instead of forcing every image into cat or dog.

## Features

- MobileNetV2 transfer learning with two-stage training (head, then fine-tuning)
- 3-class softmax output: `cat`, `dog`, `unknown`
- FastAPI service with a built-in upload page and Swagger docs
- Confidence threshold plus an optional ImageNet gate to reject non-pet images
- Every `unknown` result includes a `reason` for easy debugging
- Docker ready

## Quick start

```bash
git clone https://github.com/<your-username>/cat-dog-unknown-classifier.git
cd cat-dog-unknown-classifier
pip install -r requirements.txt
uvicorn main:app --port 8000
```

Then open http://localhost:8000 and upload an image.

> The trained model (`models/cat_dog_model.keras`) must exist before starting the server. Train it with the notebook (see below) or download it from the project's Releases page.

## Project structure

```
.
├── main.py                                      # FastAPI service
├── test_images.py                               # send a folder of images to the API
├── cat_dog_unknown_classification_cnn.ipynb     # training notebook
├── requirements.txt
├── Dockerfile
├── .gitignore
└── models/
    └── cat_dog_model.keras                      # created by the notebook
```

## Model

| Item | Value |
|---|---|
| Base model | MobileNetV2 (ImageNet weights), top 30 layers fine-tuned |
| Input | 160 x 160 x 3, raw 0-255 pixels (rescaling and augmentation are inside the model) |
| Output | `Dense(3, softmax)`: `0 = cat`, `1 = dog`, `2 = unknown` |
| Loss | `sparse_categorical_crossentropy` |
| Cat / dog data | Kaggle: `vishallazrus/cat-vs-dog-image-classification-making-prediction` |
| Unknown data | Kaggle: `alessiocorrado99/animals10` (horse, cow, sheep, elephant, squirrel, chicken, butterfly, spider) |

## Training

1. Open `cat_dog_unknown_classification_cnn.ipynb` in Colab or Jupyter.
2. Run all cells. It downloads both datasets, trains, and prints a classification report for all three classes.
3. Put the saved `cat_dog_model.keras` in the `models/` folder.

If accuracy is low: use `IMG = 224`, train 8+8 epochs, and keep the unknown set about the same size as each other class.

## API

### `POST /predict`

```bash
curl -X POST -F "file=@cat.jpg" http://localhost:8000/predict
```

```json
{
  "label": "cat",
  "confidence": 0.9731,
  "probabilities": {"cat": 0.9731, "dog": 0.0184, "unknown": 0.0085}
}
```

When the label is `unknown`, a `reason` field is included:

- `model predicted the unknown class`
- `confidence X below CONF_THRESHOLD Y`
- `not recognised as a cat or dog` (only when the ImageNet gate is on)

Errors: `400` invalid or empty image, `413` file too large.

### `GET /health`

Returns whether the model is loaded and the active settings.

Interactive docs are at `/docs`.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `models/cat_dog_model.keras` | Path to the saved model |
| `CONF_THRESHOLD` | `0.45` | Top class below this probability becomes `unknown` |
| `CLASS_NAMES` | `cat,dog,unknown` | Class order; must match training labels |
| `USE_IMAGENET_GATE` | `0` | `1` adds an extra ImageNet check that the image is a cat or dog |
| `GATE_THRESHOLD` | `0.70` | Minimum summed cat/dog probability for the gate |
| `MAX_UPLOAD_MB` | `10` | Maximum upload size |

## Test a folder of images

```bash
python test_images.py path/to/images
```

## Docker

```bash
docker build -t cat-dog-unknown .
docker run -p 8000:7860 cat-dog-unknown
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `Model file not found` | Put the model at `models/cat_dog_model.keras` or set `MODEL_PATH` |
| Real cats or dogs return `unknown` | Read the `reason`: gate, then set `USE_IMAGENET_GATE=0`; threshold, then lower `CONF_THRESHOLD`; model, then retrain with more varied cat/dog images |
| Horse or other animal returns `dog` | Add more unknown examples, or set `USE_IMAGENET_GATE=1` |
| Wolf / fox returns `dog` | Add wolves, foxes and similar animals to the unknown training data |
| `CERTIFICATE_VERIFY_FAILED` on download | Set `SSL_CERT_FILE` to `certifi.where()` |
| Upload fails with a form error | Install `python-multipart` |

## Known limitations

- The model only knows what it saw in training; unseen animals or objects can still be called cat or dog.
- Photos with people, heavy clutter, or tiny pets are harder than clean pet photos.
- Wild cats and dog-like animals need to be in the unknown data to be rejected reliably.

## License

MIT. See `LICENSE`.
