"""
Deepfake Detection Backend — FastAPI

Routes:
GET  /
GET  /health
POST /analyze/audio  -> predict_audio()
POST /analyze/image  -> predict_image()
POST /analyze/video  -> predict_video()
"""

import os
import tempfile
import traceback

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models.audio_model import predict_audio
from models.image_model import predict_image
from models.video_model import predict_video


app = FastAPI(title="Deepfake Detection API", version="1.0.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================
# Helper: Save uploaded file temporarily
# ==========================================================

async def save_upload_file(file: UploadFile):
    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Empty file received")

    suffix = os.path.splitext(file.filename or "")[1]

    if suffix == "":
        suffix = ".tmp"

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(content)
    temp_file.close()

    return temp_file.name


# ==========================================================
# Basic Routes
# ==========================================================

@app.get("/")
def home():
    return {
        "message": "Deepfake Detection Backend Running",
        "available_routes": {
            "health": "/health",
            "audio": "/analyze/audio",
            "image": "/analyze/image",
            "video": "/analyze/video",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health():
    device = "CUDA" if torch.cuda.is_available() else "CPU"

    return {
        "status": "ok",
        "device": device,
        "message": "API is running successfully"
    }


# ==========================================================
# Audio Detection Route
# ==========================================================

@app.post("/analyze/audio")
async def analyze_audio(file: UploadFile = File(...)):
    temp_path = None

    try:
        temp_path = await save_upload_file(file)

        # This connects your uploaded audio to backend/models/audio_model.py
        result = predict_audio(temp_path)

        return result

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Audio analysis failed: {str(e)}"
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ==========================================================
# Image Detection Route
# ==========================================================

@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    temp_path = None

    try:
        temp_path = await save_upload_file(file)

        # This connects your uploaded image to backend/models/image_model.py
        result = predict_image(temp_path)

        return result

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Image analysis failed: {str(e)}"
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ==========================================================
# Video / Lip Sync Detection Route
# ==========================================================

@app.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    temp_path = None

    try:
        temp_path = await save_upload_file(file)

        # This connects your uploaded video to backend/models/video_model.py
        result = predict_video(temp_path)

        return result

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Video analysis failed: {str(e)}"
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)