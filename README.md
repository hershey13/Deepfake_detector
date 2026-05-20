# Deepfake Detection Web App

A full-stack deepfake detection prototype that analyzes uploaded **images, audio, and video files** to identify possible manipulation patterns. The project combines a responsive web interface with a FastAPI backend and machine learning-based detection modules for media analysis.

## Live Demo

Frontend: https://deepfake-detection-snowy.vercel.app/

## About the Project

Deepfake Detection Web App is designed to help users upload media files and receive a prediction indicating whether the content appears **real**, **fake**, or **suspicious**.

The system supports multiple media formats and provides signal-based analysis for different types of deepfake detection, including image artifacts, audio irregularities, and video/lip-sync inconsistencies.

## Features

- Upload and analyze image, audio, and video files
- Image deepfake detection using pixel-level and artifact-based analysis
- Audio deepfake detection using spectral and MFCC-based features
- Video/lip-sync analysis using frame-based motion patterns
- FastAPI backend for handling model inference
- Responsive frontend with clean user interface
- Analysis history for previously tested files
- Export results in JSON and CSV format
- Backend health-check endpoint to verify model availability

## Tech Stack

### Frontend
- HTML
- CSS
- JavaScript
- Vercel for deployment

### Backend
- Python
- FastAPI
- Uvicorn
- PyTorch
- TorchVision
- TorchAudio
- OpenCV
- Librosa
- NumPy
- Pillow

### Machine Learning
- Image classification model
- Audio deepfake detection model
- Video/lip-sync detection model
- Model weights loaded from backend/weights
