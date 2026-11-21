"""
VocalSense AI - RAVDESS Dataset Loader, Validator, and EDA Utility
"""

import os
import sys
import zipfile
import shutil
import urllib.request
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import soundfile as sf
from tqdm import tqdm

# Import project configuration
from ml.config import (
    RAVDESS_DIR,
    PROCESSED_DIR,
    OUTPUTS_DIR,
    RAVDESS_EMOTION_MAP,
    EMOTIONS,
    EMOTION_DISPLAY_NAMES,
    EMOTION_COLORS,
)

ZENODO_RAVDESS_SPEECH_URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"


class TqdmUpTo(tqdm):
    """Provides progress bar for urllib download."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_and_extract_ravdess(
    target_dir: Path = RAVDESS_DIR,
    url: str = ZENODO_RAVDESS_SPEECH_URL,
    force_download: bool = False
) -> Path:
    """
    Downloads and extracts the RAVDESS Speech audio dataset into target_dir.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    existing_wavs = list(target_dir.rglob("*.wav"))
    if len(existing_wavs) >= 1440 and not force_download:
        print(f"[Dataset] RAVDESS already downloaded: {len(existing_wavs)} audio files detected.")
        return target_dir

    zip_path = target_dir.parent / "Audio_Speech_Actors_01-24.zip"

    if not zip_path.exists() or force_download:
        print(f"[Dataset] Downloading RAVDESS Speech Audio from Zenodo...")
        print(f"URL: {url}")
        with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading RAVDESS") as t:
            urllib.request.urlretrieve(url, filename=str(zip_path), reporthook=t.update_to)
        print(f"[Dataset] Download complete: {zip_path.stat().st_size / (1024*1024):.2f} MB")

    print(f"[Dataset] Extracting {zip_path.name} to {target_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

    # Some zip archives extract directly into subfolders (Actor_01, etc.)
    wav_count = len(list(target_dir.rglob("*.wav")))
    print(f"[Dataset] Extraction complete: {wav_count} WAV audio files verified.")
    return target_dir


def parse_ravdess_filename(filepath: Path) -> dict:
    """
    Parses a RAVDESS filename according to the official specification:
    Filename: Modality-Channel-Emotion-Intensity-Statement-Repetition-Actor.wav
    Example: 03-01-06-01-02-01-12.wav
    """
    stem = filepath.stem
    parts = stem.split("-")
    if len(parts) != 7:
        return None

    modality = parts[0]      # 01 = full-AV, 02 = video-only, 03 = audio-only
    channel = parts[1]       # 01 = speech, 02 = song
    emotion_code = parts[2]  # 01 to 08
    intensity_code = parts[3]# 01 = normal, 02 = strong
    statement_code = parts[4]# 01 = "Kids...", 02 = "Dogs..."
    repetition_code = parts[5]# 01 = 1st, 02 = 2nd
    actor_id = int(parts[6]) # 01 to 24

    # Determine gender: Odd = Male, Even = Female
    gender = "male" if actor_id % 2 != 0 else "female"

    emotion = RAVDESS_EMOTION_MAP.get(emotion_code, None)
    intensity = "normal" if intensity_code == "01" else "strong"
    statement = "Kids are talking by the door" if statement_code == "01" else "Dogs are sitting by the door"

    return {
        "filepath": str(filepath.resolve()),
        "filename": filepath.name,
        "modality": modality,
        "channel": channel,
        "emotion_code": emotion_code,
        "emotion": emotion,
        "intensity": intensity,
        "statement": statement,
        "repetition": int(repetition_code),
        "actor_id": actor_id,
        "gender": gender
    }


def validate_and_catalog_dataset(dataset_dir: Path = RAVDESS_DIR) -> pd.DataFrame:
    """
    Scans dataset_dir, parses metadata, validates audio integrity with soundfile,
    and returns a structured pandas DataFrame.
    """
    audio_files = list(dataset_dir.rglob("*.wav"))
    if not audio_files:
        raise FileNotFoundError(f"No WAV files found in {dataset_dir}")

    records = []
    corrupted_count = 0

    print(f"[Validator] Validating {len(audio_files)} audio files...")
    for f in tqdm(audio_files, desc="Validating files"):
        parsed = parse_ravdess_filename(f)
        if not parsed or parsed["emotion"] is None:
            continue

        # Check audio readability and retrieve audio signal metrics
        try:
            info = sf.info(str(f))
            if info.frames == 0 or info.duration <= 0:
                print(f"Warning: Empty audio file detected {f.name}")
                corrupted_count += 1
                continue

            parsed["duration"] = round(info.duration, 3)
            parsed["sample_rate"] = info.samplerate
            parsed["channels"] = info.channels
            parsed["frames"] = info.frames
            records.append(parsed)
        except Exception as e:
            print(f"Warning: Corrupted audio file {f.name}: {e}")
            corrupted_count += 1

    df = pd.DataFrame(records)
    print(f"[Validator] Cataloged {len(df)} valid recordings ({corrupted_count} corrupted or skipped).")

    # Export catalog to CSV
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = PROCESSED_DIR / "ravdess_metadata.csv"
    df.to_csv(out_csv, index=False)
    print(f"[Validator] Saved metadata catalog to {out_csv}")
    return df


def generate_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Computes summary metrics for the dataset.
    """
    summary = {
        "total_samples": len(df),
        "unique_actors": df["actor_id"].nunique(),
        "male_actors": df[df["gender"] == "male"]["actor_id"].nunique(),
        "female_actors": df[df["gender"] == "female"]["actor_id"].nunique(),
        "samples_per_gender": df["gender"].value_counts().to_dict(),
        "samples_per_emotion": df["emotion"].value_counts().to_dict(),
        "duration_min": float(df["duration"].min()),
        "duration_max": float(df["duration"].max()),
        "duration_mean": round(float(df["duration"].mean()), 2),
        "duration_std": round(float(df["duration"].std()), 2),
    }
    return summary


def plot_exploratory_visualizations(df: pd.DataFrame, output_dir: Path = OUTPUTS_DIR):
    """
    Generates and saves exploratory data analysis plots:
    1. Emotion distribution
    2. Gender vs Emotion distribution
    3. Audio duration distribution
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="darkgrid")
    plt.rcParams.update({
        "figure.facecolor": "#070b14",
        "axes.facecolor": "#0c1222",
        "text.color": "#f1f5f9",
        "axes.labelcolor": "#cbd5e1",
        "xtick.color": "#94a3b8",
        "ytick.color": "#94a3b8",
        "grid.color": "#1e293b"
    })

    # Plot 1: Samples per Emotion
    fig, ax = plt.subplots(figsize=(10, 5))
    emotion_order = [e for e in EMOTIONS if e in df["emotion"].values]
    palette = [EMOTION_COLORS.get(e, "#3b82f6") for e in emotion_order]

    sns.countplot(
        data=df,
        x="emotion",
        order=emotion_order,
        palette=palette,
        ax=ax
    )
    ax.set_title("VocalSense AI — RAVDESS Audio Samples per Emotion Class", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Emotion Class", fontsize=11, labelpad=10)
    ax.set_ylabel("Number of Audio Recordings", fontsize=11, labelpad=10)

    # Annotate bars
    for p in ax.patches:
        height = int(p.get_height())
        ax.annotate(f"{height}", (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=10, color='#f1f5f9', xytext=(0, 4),
                    textcoords='offset points')

    plt.tight_layout()
    chart1_path = output_dir / "eda_emotion_distribution.png"
    plt.savefig(chart1_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved emotion distribution plot to {chart1_path}")

    # Plot 2: Gender vs Emotion Breakdown
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.countplot(
        data=df,
        x="emotion",
        hue="gender",
        order=emotion_order,
        palette={"male": "#38bdf8", "female": "#d946ef"},
        ax=ax
    )
    ax.set_title("VocalSense AI — Emotion Distribution by Speaker Gender (Male vs Female)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Emotion Class", fontsize=11, labelpad=10)
    ax.set_ylabel("Count", fontsize=11, labelpad=10)
    ax.legend(title="Gender", framealpha=0.2, loc="upper right")

    plt.tight_layout()
    chart2_path = output_dir / "eda_gender_distribution.png"
    plt.savefig(chart2_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved gender breakdown plot to {chart2_path}")

    # Plot 3: Duration Distribution
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.histplot(
        df["duration"],
        kde=True,
        color="#3b82f6",
        bins=25,
        ax=ax
    )
    ax.axvline(df["duration"].mean(), color="#ef4444", linestyle="--", label=f"Mean ({df['duration'].mean():.2f}s)")
    ax.set_title("VocalSense AI — Audio Duration Distribution", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Duration (seconds)", fontsize=11, labelpad=10)
    ax.set_ylabel("Frequency", fontsize=11, labelpad=10)
    ax.legend(framealpha=0.2)

    plt.tight_layout()
    chart3_path = output_dir / "eda_duration_distribution.png"
    plt.savefig(chart3_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved duration distribution plot to {chart3_path}")


if __name__ == "__main__":
    download_and_extract_ravdess()
    df = validate_and_catalog_dataset()
    summary = generate_dataset_summary(df)
    print("\n--- DATASET SUMMARY ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    plot_exploratory_visualizations(df)
    print("\n[Done] Dataset exploration completed successfully.")
