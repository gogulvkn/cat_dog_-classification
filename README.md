# Cat / Dog classifier – deployment

```
cat_dog_deploy/
├── app/main.py            FastAPI service
├── models/cat_dog_model.keras   <- put your trained model here
├── requirements.txt
├── Dockerfile / docker-compose.yml
└── test_client.py
```

## 1. Add the model
Download `cat_dog_model.keras` from Colab and place it in `models/`.

## 2. Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## 3. Or Docker
```bash
docker compose up --build
```

## 4. Test
```bash
curl -F "file=@cat.jpg" http://localhost:8000/predict
python test_client.py horse.jpg
```
Docs UI: http://localhost:8000/docs

Response: `{"label": "cat|dog|unknown", "confidence": 0.98, "probabilities": {...}, "cat_dog_score": 0.97}`

## Config (env vars)
| Var | Default | Meaning |
|---|---|---|
| MODEL_PATH | models/cat_dog_model.keras | model file |
| CONF_THRESHOLD | 0.80 | below this -> "unknown" |
| USE_IMAGENET_GATE | 1 | pretrained check that the image is a cat/dog (fixes horse -> dog) |
| GATE_THRESHOLD | 0.50 | min cat+dog ImageNet probability |
| MAX_UPLOAD_MB | 10 | upload limit |

Use the same TensorFlow/Keras major version you trained with in Colab (`!pip show tensorflow`) and pin it in `requirements.txt`.
