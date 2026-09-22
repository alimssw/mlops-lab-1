import io
import os
from contextlib import asynccontextmanager

import mlflow
import mlflow.pyfunc
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from .data import CATEGORIES, IMAGE_SIZE

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_URI = "models:/food11@champion"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    app.state.model = mlflow.pyfunc.load_model(MODEL_URI)
    meta = app.state.model.metadata
    print(f"loaded {MODEL_URI} from {MLFLOW_TRACKING_URI} (model_id={meta.model_id}, run_id={meta.run_id})", flush=True)
    yield


app = FastAPI(lifespan=lifespan)


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMAGE_SIZE, Image.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - MEAN) / STD
    return array.transpose(2, 0, 1)[np.newaxis]


def softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    try:
        image = Image.open(io.BytesIO(await file.read()))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="file is not an image")

    logits = app.state.model.predict(preprocess(image))[0]
    probs = softmax(np.asarray(logits, dtype=np.float32))
    index = int(probs.argmax())
    return {"category": CATEGORIES[index], "confidence": round(float(probs[index]), 4)}
