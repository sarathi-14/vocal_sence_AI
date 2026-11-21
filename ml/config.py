"""
VocalSense AI - Machine Learning Configuration & Constants
"""

import os
from pathlib import Path

# Base Paths
ML_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ML_DIR.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
RAVDESS_DIR = DATASET_DIR / "ravdess"
TESS_DIR = DATASET_DIR / "tess"
PROCESSED_DIR = DATASET_DIR / "processed"
SAVED_MODELS_DIR = ML_DIR / "saved_models"
OUTPUTS_DIR = ML_DIR / "outputs"

# Audio Preprocessing Parameters
SAMPLE_RATE = 22050        # Consistent sampling rate (Hz)
AUDIO_DURATION = 3.0       # Target duration in seconds for fixed-length input
TARGET_SAMPLES = int(SAMPLE_RATE * AUDIO_DURATION)  # 66,150 samples

# Feature Extraction Parameters
N_MFCC = 40                # Number of MFCC coefficients
N_FFT = 2048               # FFT window size
HOP_LENGTH = 512           # Hop length for STFT
N_MELS = 128               # Mel filterbanks

# Emotion Taxonomy - 8 Target Emotions
EMOTIONS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised"
]

# Title Case Formats for UI Presentation
EMOTION_DISPLAY_NAMES = {
    "neutral": "Neutral",
    "calm": "Calm",
    "happy": "Happy",
    "sad": "Sad",
    "angry": "Angry",
    "fearful": "Fearful",
    "disgust": "Disgust",
    "surprised": "Surprised"
}

# Emotion Color Palette for UI and Charts
EMOTION_COLORS = {
    "neutral": "#94a3b8",   # Slate / Gray
    "calm": "#38bdf8",      # Sky Blue
    "happy": "#facc15",      # Yellow
    "sad": "#60a5fa",        # Soft Blue
    "angry": "#ef4444",      # Red
    "fearful": "#a855f7",    # Purple
    "disgust": "#10b981",    # Emerald Green
    "surprised": "#f97316"   # Orange
}

# RAVDESS Dataset Code Mapping
# Format: 03-01-XX-01-01-01-01.wav
# XX: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Model Training Hyperparameters
RANDOM_SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 60

# Ensure essential output and model directories exist
for path in [PROCESSED_DIR, SAVED_MODELS_DIR, OUTPUTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)
