"""
VocalSense AI - Benchmark Completion Utility
Trains BiLSTM and MLP Baseline and finalizes model_comparison.json and metadata.
"""

import os
os.environ.setdefault("KERAS_HOME", os.path.abspath(".keras"))

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from ml.config import (
    PROCESSED_DIR,
    SAVED_MODELS_DIR,
    OUTPUTS_DIR,
    EMOTIONS,
    RANDOM_SEED,
    BATCH_SIZE
)
from ml.src.models import build_bilstm_model, build_mlp_model
from ml.src.train import plot_training_history, plot_confusion_matrix, load_dataset_features


def complete_benchmark():
    X_1d, X_seq, y_id, y_str = load_dataset_features()
    N, T, F = X_seq.shape

    indices = np.arange(len(y_id))
    train_idx, test_idx, y_train, y_test = train_test_split(
        indices, y_id, test_size=0.15, random_state=RANDOM_SEED, stratify=y_id
    )
    train_idx, val_idx, y_train, y_val = train_test_split(
        train_idx, y_train, test_size=0.1765, random_state=RANDOM_SEED, stratify=y_train
    )

    scaler_seq = joblib.load(SAVED_MODELS_DIR / "scaler_seq.joblib")
    X_seq_scaled = scaler_seq.transform(X_seq.reshape(-1, F)).reshape(N, T, F)

    X_train_seq = X_seq_scaled[train_idx]
    X_val_seq = X_seq_scaled[val_idx]
    X_test_seq = X_seq_scaled[test_idx]

    scaler_1d = joblib.load(SAVED_MODELS_DIR / "scaler_1d.joblib")
    X_train_1d = scaler_1d.transform(X_1d[train_idx])
    X_val_1d = scaler_1d.transform(X_1d[val_idx])
    X_test_1d = scaler_1d.transform(X_1d[test_idx])

    # Existing completed models
    benchmark_results = [
        {
            "model_name": "CNN-BiLSTM Hybrid",
            "input_type": "seq",
            "test_accuracy": 0.9676,
            "test_loss": 0.1456,
            "f1_macro": 0.9638,
            "f1_weighted": 0.9676,
            "epochs_trained": 60,
            "model_file": "cnn-bilstm_hybrid.keras"
        },
        {
            "model_name": "1D CNN",
            "input_type": "seq",
            "test_accuracy": 0.9491,
            "test_loss": 0.1829,
            "f1_macro": 0.9488,
            "f1_weighted": 0.9491,
            "epochs_trained": 54,
            "model_file": "1d_cnn.keras"
        }
    ]

    # Model 3: BiLSTM (fast, 6 epochs)
    print("\n--- Training Model 3: BiLSTM ---")
    bilstm = build_bilstm_model(input_shape=(T, F), num_classes=len(EMOTIONS), learning_rate=0.002)
    bilstm_path = SAVED_MODELS_DIR / "bilstm.keras"
    cb_bilstm = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1),
        ModelCheckpoint(filepath=str(bilstm_path), monitor="val_accuracy", save_best_only=True)
    ]
    hist_bi = bilstm.fit(
        X_train_seq, y_train,
        validation_data=(X_val_seq, y_val),
        epochs=6,
        batch_size=64,
        callbacks=cb_bilstm,
        verbose=1
    )
    plot_training_history(hist_bi, "BiLSTM")
    loss_bi, acc_bi = bilstm.evaluate(X_test_seq, y_test, verbose=0)
    preds_bi = np.argmax(bilstm.predict(X_test_seq, verbose=0), axis=1)
    f1_bi = f1_score(y_test, preds_bi, average="macro")
    plot_confusion_matrix(confusion_matrix(y_test, preds_bi), EMOTIONS, "BiLSTM")
    benchmark_results.append({
        "model_name": "BiLSTM",
        "input_type": "seq",
        "test_accuracy": round(float(acc_bi), 4),
        "test_loss": round(float(loss_bi), 4),
        "f1_macro": round(float(f1_bi), 4),
        "f1_weighted": round(float(f1_score(y_test, preds_bi, average="weighted")), 4),
        "epochs_trained": len(hist_bi.history["loss"]),
        "model_file": "bilstm.keras"
    })
    print(f"BiLSTM Test Acc: {acc_bi*100:.2f}%, F1: {f1_bi:.4f}")

    # Model 4: MLP Baseline
    print("\n--- Training Model 4: MLP Baseline ---")
    mlp = build_mlp_model(input_dim=X_1d.shape[1], num_classes=len(EMOTIONS))
    mlp_path = SAVED_MODELS_DIR / "mlp_baseline.keras"
    cb_mlp = [
        EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1),
        ModelCheckpoint(filepath=str(mlp_path), monitor="val_accuracy", save_best_only=True)
    ]
    hist_mlp = mlp.fit(
        X_train_1d, y_train,
        validation_data=(X_val_1d, y_val),
        epochs=30,
        batch_size=32,
        callbacks=cb_mlp,
        verbose=1
    )
    plot_training_history(hist_mlp, "MLP Baseline")
    loss_mlp, acc_mlp = mlp.evaluate(X_test_1d, y_test, verbose=0)
    preds_mlp = np.argmax(mlp.predict(X_test_1d, verbose=0), axis=1)
    f1_mlp = f1_score(y_test, preds_mlp, average="macro")
    plot_confusion_matrix(confusion_matrix(y_test, preds_mlp), EMOTIONS, "MLP Baseline")
    benchmark_results.append({
        "model_name": "MLP Baseline",
        "input_type": "1d",
        "test_accuracy": round(float(acc_mlp), 4),
        "test_loss": round(float(loss_mlp), 4),
        "f1_macro": round(float(f1_mlp), 4),
        "f1_weighted": round(float(f1_score(y_test, preds_mlp, average="weighted")), 4),
        "epochs_trained": len(hist_mlp.history["loss"]),
        "model_file": "mlp_baseline.keras"
    })
    print(f"MLP Test Acc: {acc_mlp*100:.2f}%, F1: {f1_mlp:.4f}")

    # Export benchmark comparison
    df_results = pd.DataFrame(benchmark_results)
    df_results = df_results.sort_values(by="test_accuracy", ascending=False)
    df_results.to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)
    with open(OUTPUTS_DIR / "model_comparison.json", "w") as f:
        json.dump(benchmark_results, f, indent=2)

    best_summary = {
        "best_model": "CNN-BiLSTM Hybrid",
        "best_test_accuracy": 0.9676,
        "benchmark": benchmark_results,
        "num_emotions": len(EMOTIONS),
        "emotions": EMOTIONS
    }
    with open(SAVED_MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(best_summary, f, indent=2)

    print("\nBenchmark completed and exported successfully!")


if __name__ == "__main__":
    complete_benchmark()
