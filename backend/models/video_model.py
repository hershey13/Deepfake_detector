import os
import cv2
import torch
import torch.nn as nn
import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "weights", "lipsync_model.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LipSyncModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.lstm = nn.LSTM(32 * 16 * 16, 64, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        b, t, c, h, w = x.size()

        x = x.view(b * t, c, h, w)
        x = self.cnn(x)
        x = x.reshape(b, t, -1)

        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)

        return x


def load_lipsync_model():
    if not os.path.exists(MODEL_PATH):
        print("Lip-sync weights not found at:", MODEL_PATH)
        return None

    model = LipSyncModel().to(device)

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)

        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        new_state_dict = {}

        for key, value in state_dict.items():
            if key.startswith("module."):
                new_key = key.replace("module.", "")
            else:
                new_key = key

            new_state_dict[new_key] = value

        model.load_state_dict(new_state_dict, strict=True)
        model.eval()

        print("Lip-sync model loaded successfully from:", MODEL_PATH)
        return model

    except Exception as e:
        print("Error loading lip-sync model:", str(e))
        return None


model = load_lipsync_model()


def extract_video_sequence(video_path, seq_len=30):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return None

    frame_indices = np.linspace(0, total_frames - 1, seq_len).astype(int)

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        h, w, _ = frame.shape

        x1 = int(w * 0.3)
        x2 = int(w * 0.7)
        y1 = int(h * 0.5)
        y2 = int(h * 0.85)

        mouth = frame[y1:y2, x1:x2]

        if mouth.size == 0:
            continue

        mouth = cv2.resize(mouth, (64, 64))
        mouth = cv2.cvtColor(mouth, cv2.COLOR_BGR2RGB)
        mouth = mouth.astype(np.float32) / 255.0

        frames.append(mouth)

    cap.release()

    if len(frames) != seq_len:
        return None

    sequence = np.array(frames, dtype=np.float32)
    sequence = np.transpose(sequence, (0, 3, 1, 2))

    return sequence


def predict_video(file_path):
    if model is None:
        return {
            "type": "video",
            "prediction": "Lip-sync model weights not loaded",
            "confidence": 0,
            "message": f"Place lipsync_model.pth inside {MODEL_PATH}"
        }

    sequence = extract_video_sequence(file_path)

    if sequence is None:
        return {
            "type": "video",
            "prediction": "Could not extract enough frames",
            "confidence": 0,
            "message": "Upload a longer or clearer video."
        }

    tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    real_prob = float(probs[0])
    fake_prob = float(probs[1])

    if pred_idx == 1:
        prediction = "Fake / Lip-sync Mismatch"
        confidence = fake_prob
    else:
        prediction = "Real / Lip-sync OK"
        confidence = real_prob

    return {
        "type": "video",
        "prediction": prediction,
        "confidence": round(confidence * 100, 2),
        "real_probability": round(real_prob * 100, 2),
        "fake_probability": round(fake_prob * 100, 2),
        "model_path": MODEL_PATH
    }