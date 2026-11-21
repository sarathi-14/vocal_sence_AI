"""
VocalSense AI - Deep Learning Model Architectures for Speech Emotion Recognition
Implements 1D CNN, BiLSTM, and Hybrid CNN-BiLSTM Architectures.
"""

import os
os.environ.setdefault("KERAS_HOME", os.path.abspath(".keras"))

from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers


def build_1d_cnn_model(
    input_shape: Tuple[int, int] = (126, 40),
    num_classes: int = 8,
    learning_rate: float = 0.001
) -> models.Model:
    """
    Multi-layer 1D Convolutional Neural Network for spectral sequence feature extraction.
    Input: (time_frames, n_mfcc) -> default (130, 40)
    """
    inputs = layers.Input(shape=input_shape, name="audio_input")

    # Conv Block 1
    x = layers.Conv1D(64, kernel_size=5, strides=1, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    # Conv Block 2
    x = layers.Conv1D(128, kernel_size=5, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)

    # Conv Block 3
    x = layers.Conv1D(256, kernel_size=3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)

    # Global representation
    x = layers.GlobalAveragePooling1D()(x)

    # Dense Classifier
    x = layers.Dense(128, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="emotion_probabilities")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="VocalSense_1D_CNN")
    opt = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_bilstm_model(
    input_shape: Tuple[int, int] = (126, 40),
    num_classes: int = 8,
    learning_rate: float = 0.001
) -> models.Model:
    """
    Bidirectional Long Short-Term Memory Network for capturing temporal emotional prosody.
    """
    inputs = layers.Input(shape=input_shape, name="audio_input")

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2))(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False, dropout=0.2))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="emotion_probabilities")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="VocalSense_BiLSTM")
    opt = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_cnn_bilstm_model(
    input_shape: Tuple[int, int] = (126, 40),
    num_classes: int = 8,
    learning_rate: float = 0.0008
) -> models.Model:
    """
    State-of-the-Art Hybrid CNN + Bidirectional LSTM Architecture:
    1. Conv1D layers extract local spectral and harmonic patterns.
    2. BiLSTM layers capture sequential acoustic contour and tone dynamics.
    3. Dense classification head with dropout and L2 weight decay.
    """
    inputs = layers.Input(shape=input_shape, name="audio_input")

    # Front-end Convolutional Feature Extractor
    x = layers.Conv1D(64, kernel_size=5, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(128, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)

    # Recurrent Sequence Modeler
    x = layers.Bidirectional(layers.LSTM(96, return_sequences=True, dropout=0.2))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False, dropout=0.2))(x)
    x = layers.BatchNormalization()(x)

    # Classification Head
    x = layers.Dense(128, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="emotion_probabilities")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="VocalSense_CNN_BiLSTM_Hybrid")
    opt = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_mlp_model(
    input_dim: int = 215,
    num_classes: int = 8,
    learning_rate: float = 0.001
) -> models.Model:
    """
    Multi-Layer Perceptron trained on aggregated 1D statistical audio features.
    """
    inputs = layers.Input(shape=(input_dim,), name="features_1d_input")

    x = layers.Dense(256, kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="emotion_probabilities")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="VocalSense_MLP_Baseline")
    opt = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
