import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Deepfake Detection API", version="1.0.0")

# ══ CORS MUST BE FIRST — before any routes ════════════════════════════════════
origins = [
    # "http://localhost:5173",
    # "http://localhost:3000",
    "https://deepfake-detection-snowy.vercel.app/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══ Safe model imports ════════════════════════════════════════════════════════
try:
    from models.audio_model import predict_audio
    AUDIO_OK = True
except Exception as e:
    print("Audio model import failed:", e)
    AUDIO_OK = False

try:
    from models.video_model import predict_video
    VIDEO_OK = True
except Exception as e:
    print("Video model import failed:", e)
    VIDEO_OK = False

try:
    from models.image_model import predict_image
    IMAGE_OK = True
except Exception as e:
    print("Image model import failed:", e)
    IMAGE_OK = False

# ══ Routes ════════════════════════════════════════════════════════════════════
@app.get("/")
def home():
    return {"message": "Deepfake Detection Backend is running"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {
            "audio": AUDIO_OK,
            "video": VIDEO_OK,
            "image": IMAGE_OK,
        }
    }

@app.post("/analyze/audio")
async def analyze_audio(file: UploadFile = File(...)):
    if not AUDIO_OK:
        raise HTTPException(status_code=503, detail="Audio model not available")

    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = predict_audio(tmp_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

@app.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    if not VIDEO_OK:
        raise HTTPException(status_code=503, detail="Video model not available")

    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = predict_video(tmp_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    if not IMAGE_OK:
        raise HTTPException(status_code=503, detail="Image model not available")

    suffix = os.path.splitext(file.filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = predict_image(tmp_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)
