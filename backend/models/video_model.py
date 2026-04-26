import os
import cv2
import torch
import torch.nn as nn
import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
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


model = LipSyncModel().to(device)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
else:
    model = None


def extract_video_sequence(video_path, seq_len=30):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        frame_indices = list(range(seq_len))
    else:
        frame_indices = np.linspace(0, total_frames - 1, seq_len).astype(int)

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        h, w, _ = frame.shape

        x1, x2 = int(w * 0.3), int(w * 0.7)
        y1, y2 = int(h * 0.5), int(h * 0.85)

        mouth = frame[y1:y2, x1:x2]

        if mouth.size == 0:
            continue

        mouth = cv2.resize(mouth, (64, 64))
        mouth = mouth / 255.0
        frames.append(mouth)

    cap.release()

    if len(frames) != seq_len:
        return None

    sequence = np.array(frames)

    # Convert from BGR to RGB-like order is not critical here,
    # but keep same numeric format.
    # Shape: T, H, W, C → T, C, H, W
    sequence = np.transpose(sequence, (0, 3, 1, 2))

    return sequence


def predict_video(file_path):
    if model is None:
        return {
            "type": "video",
            "prediction": "Lip-sync model weights not found",
            "confidence": 0,
            "message": "Place lipsync_model.pth inside backend/weights/"
        }

    sequence = extract_video_sequence(file_path)

    if sequence is None:
        return {
            "type": "video",
            "prediction": "Could not extract enough frames",
            "confidence": 0,
            "message": "Upload a longer/clearer video."
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
        "fake_probability": round(fake_prob * 100, 2)
    }