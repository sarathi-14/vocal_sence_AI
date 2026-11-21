"""
VocalSense AI - Backend Pydantic Schemas for API Requests & Responses
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class EmotionProbability(BaseModel):
    emotion: str
    display_name: str
    probability: float
    percentage: float
    color: str


class AudioDiagnostics(BaseModel):
    duration_seconds: float
    sample_rate: int
    mean_rms_energy: float
    mean_zero_crossing_rate: float
    spectral_centroid_hz: float


class EmotionInsight(BaseModel):
    primary_emotion: str
    valence: str          # Positive, Negative, Neutral
    arousal: str          # High, Moderate, Low
    description: str
    confidence_tier: str  # High Confidence (>70%), Moderate (40-70%), Low (<40%)


class EmotionPredictionResponse(BaseModel):
    success: bool
    predicted_emotion: str
    display_name: str
    confidence: float
    confidence_percentage: float
    color: str
    probabilities: List[EmotionProbability]
    diagnostics: AudioDiagnostics
    insight: EmotionInsight
    model_name: str
    inference_time_ms: float


class SegmentEmotion(BaseModel):
    segment_index: int
    start_seconds: float
    end_seconds: float
    dominant_emotion: str
    confidence: float
    color: str


class EmotionTrajectoryResponse(BaseModel):
    success: bool
    total_duration: float
    overall_emotion: str
    overall_confidence: float
    trajectory: List[SegmentEmotion]
    diagnostics: AudioDiagnostics
    inference_time_ms: float


class ModelMetrics(BaseModel):
    model_name: str
    input_type: str
    test_accuracy: float
    test_loss: float
    f1_macro: float
    f1_weighted: float
    epochs_trained: int


class BenchmarkResponse(BaseModel):
    best_model: str
    best_accuracy: float
    models: List[ModelMetrics]
    emotions: List[str]


class ModelInfoResponse(BaseModel):
    model_name: str
    best_accuracy: float
    sample_rate: int
    audio_duration: float
    input_shape: List[int]
    emotions: List[str]
    device: str
    status: str
