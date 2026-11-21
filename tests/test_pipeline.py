"""
VocalSense AI - End-to-End Automated Test Suite
Validates audio DSP preprocessing, feature extraction dimensions, model inference, and API route functions.
"""

import os
os.environ.setdefault("KERAS_HOME", os.path.abspath(".keras"))

import io
import unittest
import numpy as np
import soundfile as sf

from ml.config import (
    SAMPLE_RATE,
    AUDIO_DURATION,
    TARGET_SAMPLES,
    N_MFCC,
    EMOTIONS,
    SAVED_MODELS_DIR,
    PROCESSED_DIR
)
from ml.src.feature_extractor import (
    preprocess_audio,
    extract_acoustic_features,
    prepare_single_audio_for_inference,
    load_audio
)
from backend.app.routes.api import (
    health_check,
    get_model_info,
    get_benchmark_comparison,
    get_sample_recordings
)
from backend.app.services.emotion_service import EmotionInferenceService


def generate_synthetic_audio(duration: float = 3.0, sr: int = SAMPLE_RATE) -> bytes:
    """Generates a valid in-memory WAV audio clip for automated testing."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    sig = sig.astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, sig, sr, format="WAV")
    buf.seek(0)
    return buf.read()


class TestVocalSenseAIPipeline(unittest.TestCase):

    def test_audio_preprocessing(self):
        """Validates that preprocessing produces exact target sample length and normalization."""
        raw = np.random.randn(50000).astype(np.float32)
        clean = preprocess_audio(raw, sr=SAMPLE_RATE, target_samples=TARGET_SAMPLES, normalize=True)

        self.assertEqual(len(clean), TARGET_SAMPLES, f"Expected {TARGET_SAMPLES} samples, got {len(clean)}")
        self.assertLessEqual(np.max(np.abs(clean)), 1.0, "Audio amplitude exceeded 1.0 after normalization")
        self.assertEqual(clean.dtype, np.float32, "Audio must be float32")

    def test_feature_extraction_dimensions(self):
        """Validates temporal sequence and 1D feature dimensions."""
        sig = np.random.randn(TARGET_SAMPLES).astype(np.float32)
        seq, feat_1d = extract_acoustic_features(sig, sr=SAMPLE_RATE, n_mfcc=N_MFCC)

        self.assertEqual(seq.ndim, 2, f"Sequence must be 2D, got shape {seq.shape}")
        self.assertEqual(seq.shape[1], N_MFCC, f"Expected {N_MFCC} MFCCs, got {seq.shape[1]}")
        self.assertEqual(feat_1d.ndim, 1, f"1D features must be vector, got shape {feat_1d.shape}")
        self.assertEqual(len(feat_1d), 215, f"Expected 215 acoustic features, got {len(feat_1d)}")

    def test_health_route(self):
        """Tests health_check() route handler."""
        data = health_check()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["service"], "VocalSense AI")
        self.assertTrue(data["model_loaded"])
        self.assertEqual(data["supported_emotions"], 8)

    def test_model_info_route(self):
        """Tests get_model_info() route handler."""
        data = get_model_info()
        self.assertEqual(data.model_name, "CNN-BiLSTM Hybrid")
        self.assertAlmostEqual(data.best_accuracy, 0.9676, places=3)
        self.assertEqual(data.sample_rate, SAMPLE_RATE)
        self.assertEqual(len(data.emotions), 8)

    def test_benchmark_route(self):
        """Tests get_benchmark_comparison() route handler."""
        data = get_benchmark_comparison()
        self.assertEqual(data["best_model"], "CNN-BiLSTM Hybrid")
        self.assertEqual(len(data["models"]), 4)
        self.assertEqual(len(data["emotions"]), 8)

    def test_sample_recordings_route(self):
        """Tests get_sample_recordings() route handler."""
        data = get_sample_recordings()
        self.assertIn("samples", data)
        self.assertGreaterEqual(len(data["samples"]), 8)

    def test_live_inference_pipeline(self):
        """Tests end-to-end model inference on synthetic audio."""
        audio_bytes = generate_synthetic_audio(duration=2.5)
        service = EmotionInferenceService.get_instance()
        res = service.predict_audio(audio_bytes)

        self.assertTrue(res["success"])
        self.assertIn(res["predicted_emotion"], EMOTIONS)
        self.assertGreaterEqual(res["confidence"], 0.0)
        self.assertLessEqual(res["confidence"], 1.0)
        self.assertEqual(len(res["probabilities"]), 8)
        self.assertIn("diagnostics", res)
        self.assertIn("duration_seconds", res["diagnostics"])
        self.assertIn("insight", res)
        self.assertEqual(res["insight"]["primary_emotion"], res["predicted_emotion"])

    def test_trajectory_analysis_pipeline(self):
        """Tests multi-window temporal trajectory analysis."""
        audio_bytes = generate_synthetic_audio(duration=4.0)
        service = EmotionInferenceService.get_instance()
        res = service.analyze_trajectory(audio_bytes, window_duration=2.0, step_duration=1.0)

        self.assertTrue(res["success"])
        self.assertGreaterEqual(res["total_duration"], 3.9)
        self.assertGreaterEqual(len(res["trajectory"]), 2)
        self.assertIn("overall_emotion", res)


if __name__ == "__main__":
    unittest.main()
