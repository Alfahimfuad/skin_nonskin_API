
import io
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "skin_non_skin_model.keras"

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}

SKIN_THRESHOLD = 0.5

app = FastAPI(
    title="DermaCheckerBD Skin Gate API",
    description=(
        "Checks whether an uploaded image appears to contain skin "
        "before forwarding it to lesion analysis."
    ),
    version="1.0.0",
)

# Comma-separated list of your website origins, e.g.
# ALLOWED_ORIGINS=https://site1.com,https://site2.com
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

model = None
IMG_HEIGHT = None
IMG_WIDTH = None


# --------------------------------------------------
# Load model
# --------------------------------------------------

@app.on_event("startup")
def load_model():
    global model, IMG_HEIGHT, IMG_WIDTH

    if not MODEL_PATH.is_file():
        raise RuntimeError(
            f"Model file not found: {MODEL_PATH}\n"
            "Place skin_non_skin_model.keras inside the models folder."
        )

    try:
        model = tf.keras.models.load_model(
            str(MODEL_PATH),
            compile=False,
        )

        input_shape = model.input_shape

        if isinstance(input_shape, list):
            raise RuntimeError(
                "Only single-input models are supported."
            )

        if len(input_shape) != 4:
            raise RuntimeError(
                f"Unexpected model input shape: {input_shape}"
            )

        if input_shape[-1] != 3:
            raise RuntimeError(
                f"Expected RGB input, got: {input_shape}"
            )

        if input_shape[1] is None or input_shape[2] is None:
            raise RuntimeError(
                f"Fixed image dimensions are required: {input_shape}"
            )

        output_shape = model.output_shape

        if isinstance(output_shape, list):
            raise RuntimeError(
                "Only single-output models are supported."
            )

        if len(output_shape) != 2 or output_shape[-1] != 1:
            raise RuntimeError(
                f"Expected one binary output, got: {output_shape}"
            )

        IMG_HEIGHT = int(input_shape[1])
        IMG_WIDTH = int(input_shape[2])

        # Identify the final output layer.
        output_layer = model.layers[-1]
        output_layer_name = output_layer.name.lower()

        print("Skin detection model loaded successfully.")
        print(f"Model path: {MODEL_PATH}")
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
        print(f"Output layer: {output_layer.name}")
        print(f"Image size: {IMG_WIDTH} x {IMG_HEIGHT}")

        if "p_non_skin" not in output_layer_name:
            print(
                "WARNING: Output layer name does not confirm "
                "that the sigmoid represents non-skin."
            )

    except Exception as exc:
        raise RuntimeError(
            f"Could not load skin detection model: {exc}"
        ) from exc


# --------------------------------------------------
# Health checks
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "service": "DermaCheckerBD Skin Detection API",
        "status": "running",
        "model_loaded": model is not None,
        "endpoint": "/api/check-skin",
        "documentation": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "not_ready",
        "model_loaded": model is not None,
    }


# --------------------------------------------------
# Skin / Non-skin prediction
# --------------------------------------------------

@app.post("/api/check-skin")
async def check_skin(file: UploadFile = File(...)):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Skin detection model is not loaded.",
        )

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Please upload a JPG, PNG, or WebP image.",
        )

    try:
        content = await file.read(MAX_FILE_SIZE + 1)

        if not content:
            raise HTTPException(
                status_code=400,
                detail="The uploaded image is empty.",
            )

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Image exceeds the 10 MB limit.",
            )

        try:
            with Image.open(io.BytesIO(content)) as source_image:
                source_image.load()
                image = source_image.convert("RGB")

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
        ):
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is not a valid image.",
            )

        # Resize to the model's expected input dimensions.
        image = image.resize((IMG_WIDTH, IMG_HEIGHT))

        # The model contains its own MobileNetV2 preprocessing layer.
        # Supply RGB pixel values in the 0-255 range.
        # Do not normalize or preprocess them a second time.
        image_array = np.asarray(image, dtype=np.float32)
        batch = np.expand_dims(image_array, axis=0)

        prediction = model.predict(batch, verbose=0)
        prediction_array = np.asarray(prediction)

        if prediction_array.size != 1:
            raise HTTPException(
                status_code=500,
                detail="Expected exactly one sigmoid output.",
            )

        raw_score = float(prediction_array.reshape(-1)[0])

        if (
            not np.isfinite(raw_score)
            or not 0.0 <= raw_score <= 1.0
        ):
            raise HTTPException(
                status_code=500,
                detail="Model returned an invalid probability.",
            )

        # IMPORTANT:
        # This mapping assumes p_non_skin really represents
        # the non_skin class. Verify it against training labels.
        non_skin_score = raw_score
        skin_score = 1.0 - non_skin_score

        is_skin = non_skin_score < SKIN_THRESHOLD

        return {
            "success": True,
            "is_skin": bool(is_skin),
            "label": "skin" if is_skin else "non_skin",
            "skin_score": round(skin_score, 6),
            "non_skin_score": round(non_skin_score, 6),
            "threshold": SKIN_THRESHOLD,
            "message": (
                "Image accepted for the next analysis stage."
                if is_skin
                else "This image was classified as non-skin."
            ),
            "notice": (
                "Research prototype only; this result is not "
                "a medical diagnosis."
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Skin detection failed: {str(exc)}",
        ) from exc

    finally:
        await file.close()
