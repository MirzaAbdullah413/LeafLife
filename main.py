import io
import logging
import os

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from model_utils import load_trained_model, predict_image

torch.set_num_threads(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("leaflife")

# 1. Create the app
app = FastAPI(title="LeafLife")

# 2. CORS: set ALLOWED_ORIGINS on Render to your Vercel URL (no trailing slash)
origins = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# 3. Load the model once at startup (absolute path so it works from any folder)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "leaf_disease_model.pth")
MAX_FILE_SIZE = 5 * 1024 * 1024      # 5 MB
MAX_PIXELS = 25_000_000              # blocks "decompression bomb" images

model, val_transformation, classes, device = load_trained_model(MODEL_PATH)
logger.info("Model loaded on %s with %d classes: %s", device, len(classes), classes)


# Health check (use "/" as Render's Health Check Path)
@app.get("/")
def health_check():
    return {"status": "ok", "num_classes": len(classes)}


# Predict the disease
@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/jpg", "image/png"):
        raise HTTPException(status_code=400, detail="Only JPG/PNG images are supported")

    contents = file.file.read(MAX_FILE_SIZE + 1)
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Image too large (max 5 MB)")

    try:
        img = Image.open(io.BytesIO(contents))
        if img.width * img.height > MAX_PIXELS:
            raise HTTPException(status_code=400, detail="Image dimensions too large")
        pil_image = img.convert("RGB")
        pil_image.thumbnail((640, 640))
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Invalid or corrupted image")

    try:
        class_name, heatmap = predict_image(
            model, pil_image, val_transformation, classes, device
        )
    except Exception:
        logger.exception("Inference failed")
        raise HTTPException(
            status_code=500, detail="Prediction failed. Please try another image."
        )

    return {"disease": class_name, "heatmap": f"data:image/jpeg;base64,{heatmap}"}