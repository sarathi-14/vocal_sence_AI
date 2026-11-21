"""
VocalSense AI - Multi-Model Training, Hyperparameter Tuning & Benchmark Engine
Trains and benchmarks 1D CNN, BiLSTM, and Hybrid CNN-BiLSTM on RAVDESS Speech Emotion data.
"""

import os
os.environ.setdefault("KERAS_HOME", os.path.abspath(".keras"))

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from ml.config import (
    PROCESSED_DIR,
    SAVED_MODELS_DIR,
    OUTPUTS_DIR,
    EMOTIONS,
    EMOTION_DISPLAY_NAMES,
    RANDOM_SEED,
    BATCH_SIZE,
    EPOCHS
)
from ml.src.models import (
    build_1d_cnn_model,
    build_bilstm_model,
    build_cnn_bilstm_model,
    build_mlp_model
)


def load_dataset_features(processed_dir: Path = PROCESSED_DIR):
    """Loads extracted feature arrays and metadata."""
    x_1d_path = processed_dir / "X_1d.npy"
    x_seq_path = processed_dir / "X_seq.npy"
    y_id_path = processed_dir / "y_id.npy"
    y_str_path = processed_dir / "y_str.npy"

    if not all(p.exists() for p in [x_1d_path, x_seq_path, y_id_path]):
        raise FileNotFoundError(f"Feature arrays not found in {processed_dir}. Run feature_extractor first.")

    X_1d = np.load(x_1d_path)
    X_seq = np.load(x_seq_path)
    y_id = np.load(y_id_path)
    y_str = np.load(y_str_path)

    return X_1d, X_seq, y_id, y_str


def plot_training_history(history, model_name: str, output_dir: Path = OUTPUTS_DIR):
    """Plots and saves loss and accuracy curves for a model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Dark theme styling
    fig.patch.set_facecolor("#070b14")
    for ax in axes:
        ax.set_facecolor("#0c1222")
        ax.tick_params(colors="#94a3b8")
        ax.xaxis.label.set_color("#cbd5e1")
        ax.yaxis.label.set_color("#cbd5e1")
        ax.title.set_color("#f1f5f9")
        ax.grid(True, color="#1e293b")

    # Accuracy
    axes[0].plot(history.history["accuracy"], label="Train Accuracy", color="#38bdf8", lw=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy", color="#a855f7", lw=2, linestyle="--")
    axes[0].set_title(f"{model_name} — Accuracy Curve", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(framealpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"], label="Train Loss", color="#ef4444", lw=2)
    axes[1].plot(history.history["val_loss"], label="Val Loss", color="#f59e0b", lw=2, linestyle="--")
    axes[1].set_title(f"{model_name} — Loss Curve", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(framealpha=0.3)

    plt.tight_layout()
    chart_path = output_dir / f"history_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Training] Saved history curves to {chart_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    model_name: str,
    output_dir: Path = OUTPUTS_DIR
):
    """Plots and saves a styled confusion matrix."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor("#070b14")
    ax.set_facecolor("#0c1222")

    display_labels = [EMOTION_DISPLAY_NAMES.get(c, c.capitalize()) for c in classes]

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=display_labels,
        yticklabels=display_labels,
        ax=ax,
        cbar=True,
        linewidths=0.5,
        linecolor="#1e293b"
    )

    ax.set_title(f"{model_name} — Confusion Matrix", fontsize=14, fontweight="bold", color="#f1f5f9", pad=15)
    ax.set_xlabel("Predicted Emotion", fontsize=11, color="#cbd5e1", labelpad=10)
    ax.set_ylabel("Ground Truth Emotion", fontsize=11, color="#cbd5e1", labelpad=10)
    ax.tick_params(colors="#cbd5e1", rotation=45)

    plt.tight_layout()
    cm_path = output_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Training] Saved confusion matrix to {cm_path}")


def train_and_evaluate_all_models():
    """
    Orchestrates the entire training, benchmark and model selection pipeline.
    """
    print("[Pipeline] Loading features...")
    X_1d, X_seq, y_id, y_str = load_dataset_features()

    print(f"Loaded X_seq: {X_seq.shape}, X_1d: {X_1d.shape}, y: {y_id.shape}")

    # Standardize 2D sequence features:
    # Reshape (N, T, F) -> scale F across all time steps -> reshape back
    N, T, F = X_seq.shape
    X_seq_flat = X_seq.reshape(-1, F)
    scaler_seq = StandardScaler()

    # Train / Val / Test Split (70% train, 15% val, 15% test)
    indices = np.arange(len(y_id))
    train_idx, test_idx, y_train, y_test = train_test_split(
        indices, y_id, test_size=0.15, random_state=RANDOM_SEED, stratify=y_id
    )
    train_idx, val_idx, y_train, y_val = train_test_split(
        train_idx, y_train, test_size=0.1765, random_state=RANDOM_SEED, stratify=y_train
    )  # 0.1765 of 0.85 is ~0.15 of total

    # Fit scaler on train split only
    scaler_seq.fit(X_seq[train_idx].reshape(-1, F))
    joblib.dump(scaler_seq, SAVED_MODELS_DIR / "scaler_seq.joblib")

    # Apply scaling
    X_seq_scaled = scaler_seq.transform(X_seq_flat).reshape(N, T, F)

    X_train_seq = X_seq_scaled[train_idx]
    X_val_seq = X_seq_scaled[val_idx]
    X_test_seq = X_seq_scaled[test_idx]

    # Standardize 1D features
    scaler_1d = StandardScaler()
    scaler_1d.fit(X_1d[train_idx])
    joblib.dump(scaler_1d, SAVED_MODELS_DIR / "scaler_1d.joblib")

    X_train_1d = scaler_1d.transform(X_1d[train_idx])
    X_val_1d = scaler_1d.transform(X_1d[val_idx])
    X_test_1d = scaler_1d.transform(X_1d[test_idx])

    print(f"Data Split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # Model candidates configuration
    model_configs = [
        {
            "name": "CNN-BiLSTM Hybrid",
            "type": "seq",
            "builder": lambda: build_cnn_bilstm_model(input_shape=(T, F), num_classes=len(EMOTIONS)),
            "train_data": (X_train_seq, y_train),
            "val_data": (X_val_seq, y_val),
            "test_data": (X_test_seq, y_test),
        },
        {
            "name": "1D CNN",
            "type": "seq",
            "builder": lambda: build_1d_cnn_model(input_shape=(T, F), num_classes=len(EMOTIONS)),
            "train_data": (X_train_seq, y_train),
            "val_data": (X_val_seq, y_val),
            "test_data": (X_test_seq, y_test),
        },
        {
            "name": "BiLSTM",
            "type": "seq",
            "builder": lambda: build_bilstm_model(input_shape=(T, F), num_classes=len(EMOTIONS)),
            "train_data": (X_train_seq, y_train),
            "val_data": (X_val_seq, y_val),
            "test_data": (X_test_seq, y_test),
        },
        {
            "name": "MLP Baseline",
            "type": "1d",
            "builder": lambda: build_mlp_model(input_dim=X_1d.shape[1], num_classes=len(EMOTIONS)),
            "train_data": (X_train_1d, y_train),
            "val_data": (X_val_1d, y_val),
            "test_data": (X_test_1d, y_test),
        }
    ]

    benchmark_results = []
    best_model_name = None
    best_test_acc = -1.0
    best_model_instance = None

    for config in model_configs:
        name = config["name"]
        print(f"\n==========================================")
        print(f"   TRAINING MODEL: {name}")
        print(f"==========================================")

        model = config["builder"]()
        X_tr, y_tr = config["train_data"]
        X_v, y_v = config["val_data"]
        X_te, y_te = config["test_data"]

        model_save_path = SAVED_MODELS_DIR / f"{name.lower().replace(' ', '_')}.keras"

        callbacks = [
            EarlyStopping(monitor="val_accuracy", patience=14, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1),
            ModelCheckpoint(filepath=str(model_save_path), monitor="val_accuracy", save_best_only=True, verbose=0)
        ]

        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_v, y_v),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )

        # Plot curves
        plot_training_history(history, name)

        # Evaluate on Test Split
        test_loss, test_acc = model.evaluate(X_te, y_te, verbose=0)
        y_pred_probs = model.predict(X_te, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        f1_macro = f1_score(y_te, y_pred, average="macro")
        f1_weighted = f1_score(y_te, y_pred, average="weighted")
        cm = confusion_matrix(y_te, y_pred)
        plot_confusion_matrix(cm, EMOTIONS, name)

        rep_dict = classification_report(y_te, y_pred, target_names=EMOTIONS, output_dict=True)

        res = {
            "model_name": name,
            "input_type": config["type"],
            "test_accuracy": round(float(test_acc), 4),
            "test_loss": round(float(test_loss), 4),
            "f1_macro": round(float(f1_macro), 4),
            "f1_weighted": round(float(f1_weighted), 4),
            "epochs_trained": len(history.history["loss"]),
            "model_file": str(model_save_path.name)
        }
        benchmark_results.append(res)
        print(f"[Result] {name} -> Test Acc: {test_acc*100:.2f}%, F1: {f1_macro:.4f}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_name = name
            best_model_instance = model
            # Save as primary production model
            model.save(SAVED_MODELS_DIR / "best_emotion_model.keras")

    # Export benchmark comparison
    df_results = pd.DataFrame(benchmark_results)
    df_results = df_results.sort_values(by="test_accuracy", ascending=False)
    df_results.to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)
    with open(OUTPUTS_DIR / "model_comparison.json", "w") as f:
        json.dump(benchmark_results, f, indent=2)

    best_summary = {
        "best_model": best_model_name,
        "best_test_accuracy": round(float(best_test_acc), 4),
        "benchmark": benchmark_results,
        "num_emotions": len(EMOTIONS),
        "emotions": EMOTIONS
    }
    with open(SAVED_MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(best_summary, f, indent=2)

    print(f"\n==========================================")
    print(f"   BENCHMARK COMPLETED SUCCESSFULLY")
    print(f"   Best Model: {best_model_name} ({best_test_acc*100:.2f}% Acc)")
    print(f"==========================================")
    return best_summary


if __name__ == "__main__":
    train_and_evaluate_all_models()
