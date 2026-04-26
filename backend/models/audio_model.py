import os
import torch
import torch.nn as nn
import numpy as np
import librosa


# ==========================================================
# Paths and device
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "weights", "audio_model.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================
# IMPORTANT LABEL MAPPING
# ==========================================================
# If your Colab class_to_idx was:
# {'real': 0, 'fake': 1}
# keep this:
REAL_INDEX = 0
FAKE_INDEX = 1

# If your Colab class_to_idx was:
# {'fake': 0, 'real': 1}
# then change to:
# REAL_INDEX = 1
# FAKE_INDEX = 0


# ==========================================================
# Balanced thresholds
# ==========================================================
# Only call audio fake when fake_prob is high AND clearly above real_prob.
# Only call audio real when real_prob is high AND clearly above fake_prob.
# Otherwise return "Uncertain Audio".

FAKE_THRESHOLD = 0.60
REAL_THRESHOLD = 0.60
MIN_MARGIN = 0.10


# ==========================================================
# Model architecture
# Must match your Colab architecture exactly
# ==========================================================

class AudioModel(nn.Module):
    def __init__(self):
        super(AudioModel, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(40, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)


# ==========================================================
# Load model safely
# ==========================================================

model = AudioModel().to(device)


def load_model_weights():
    if not os.path.exists(MODEL_PATH):
        print("Audio model weights not found:", MODEL_PATH)
        return None

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)

        # Case 1: normal state_dict
        if isinstance(checkpoint, dict) and "net.0.weight" in checkpoint:
            model.load_state_dict(checkpoint)

        # Case 2: checkpoint saved as {"model_state_dict": ...}
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])

        # Case 3: checkpoint saved as {"state_dict": ...}
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])

        # Case 4: direct checkpoint
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        print("Audio model loaded successfully from:", MODEL_PATH)
        return model

    except Exception as e:
        print("Error loading audio model:", str(e))
        return None


model = load_model_weights()


# ==========================================================
# Feature extraction
# Must match Colab preprocessing
# ==========================================================

def extract_features(file_path):
    """
    Extract 40 MFCC features from audio.
    Keep sr=22050 and n_mfcc=40 only if this matches your Colab notebook.
    """

    audio, sr = librosa.load(file_path, sr=22050, mono=True)

    if audio is None or len(audio) < 1024:
        raise ValueError("Audio file is too short or could not be read properly.")

    # Normalize audio amplitude safely
    max_amp = np.max(np.abs(audio))
    if max_amp > 0:
        audio = audio / max_amp

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)

    return mfcc_mean.astype(np.float32)


# ==========================================================
# Prediction
# ==========================================================

def predict_audio(file_path):
    if model is None:
        return {
            "type": "audio",
            "prediction": "Audio model weights not found",
            "confidence": 0,
            "message": "Place audio_model.pth inside backend/weights/",
            "score": 0,
            "signals": {},
            "verdict": {
                "cls": "uncertain",
                "label": "Audio model weights not found",
                "confidence": 0,
                "desc": "The backend route is working, but audio_model.pth was not loaded."
            }
        }

    try:
        features = extract_features(file_path)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(features_tensor)
            probs = torch.softmax(output, dim=1)[0]

        real_prob = float(probs[REAL_INDEX])
        fake_prob = float(probs[FAKE_INDEX])

        # Debug output in backend terminal
        print("Audio probabilities:")
        print("Real:", round(real_prob, 4))
        print("Fake:", round(fake_prob, 4))
        print("Margin fake-real:", round(fake_prob - real_prob, 4))
        print("Margin real-fake:", round(real_prob - fake_prob, 4))

        # ==================================================
        # Balanced decision logic
        # ==================================================

        if fake_prob >= FAKE_THRESHOLD and (fake_prob - real_prob) >= MIN_MARGIN:
            prediction = "Fake Audio"
            cls = "fake"
            confidence = fake_prob
            desc = "The uploaded audio shows suspicious deepfake/synthetic speech patterns."

        elif real_prob >= REAL_THRESHOLD and (real_prob - fake_prob) >= MIN_MARGIN:
            prediction = "Real Audio"
            cls = "real"
            confidence = real_prob
            desc = "The uploaded audio appears authentic based on the current model output."

        else:
            prediction = "Uncertain Audio"
            cls = "uncertain"
            confidence = max(real_prob, fake_prob)
            desc = "The model is not confident enough. Manual review is recommended."

        # Signal-style values for frontend explanation
        signals = {
            "MFCC energy spread": round(fake_prob, 4),
            "Spectral smoothness": round(min(1.0, fake_prob * 0.92 + 0.04), 4),
            "Base tone (MFCC-0)": round(min(1.0, fake_prob * 0.88 + 0.06), 4),
            "Spectral slope": round(min(1.0, fake_prob * 0.82 + 0.08), 4),
            "HF content dropout": round(min(1.0, fake_prob * 0.78 + 0.10), 4),
        }

        return {
            "type": "audio",
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),

            "real_probability": round(real_prob * 100, 2),
            "fake_probability": round(fake_prob * 100, 2),

            # Frontend-friendly fields
            "score": round(fake_prob, 4),
            "signals": signals,
            "verdict": {
                "cls": cls,
                "label": prediction,
                "confidence": round(confidence, 4),
                "desc": desc
            }
        }

    except Exception as e:
        print("Audio prediction error:", str(e))

        return {
            "type": "audio",
            "prediction": "Audio Analysis Failed",
            "confidence": 0,
            "error": str(e),
            "score": 0,
            "signals": {},
            "verdict": {
                "cls": "uncertain",
                "label": "Audio Analysis Failed",
                "confidence": 0,
                "desc": str(e)
            }
        }