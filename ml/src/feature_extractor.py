"""
VocalSense AI - High-Performance Audio Preprocessing & Feature Extraction Engine
Pure NumPy/SciPy acoustic signal processing optimized for speed, reliability, and precision.
Extracts 40-MFCC temporal sequences and 215-dimensional multi-feature acoustic vectors.
"""

import os
import io
import json
from pathlib import Path
from typing import Tuple, Dict, Any, Union, Optional
import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal
import scipy.fftpack
from tqdm import tqdm

from ml.config import (
    SAMPLE_RATE,
    AUDIO_DURATION,
    TARGET_SAMPLES,
    N_MFCC,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    EMOTIONS,
    PROCESSED_DIR,
    SAVED_MODELS_DIR,
)


def load_audio(
    audio_source: Union[str, Path, io.BytesIO, bytes, np.ndarray],
    target_sr: int = SAMPLE_RATE,
) -> Tuple[np.ndarray, int]:
    """
    Robust audio loader supporting file paths, byte buffers, and numpy arrays.
    Returns (signal, sample_rate).
    """
    if isinstance(audio_source, np.ndarray):
        signal = audio_source.astype(np.float32)
        sr = target_sr
    else:
        if isinstance(audio_source, bytes):
            audio_source = io.BytesIO(audio_source)

        if isinstance(audio_source, io.BytesIO):
            audio_source.seek(0)
            signal, sr = sf.read(audio_source, dtype="float32")
        else:
            signal, sr = sf.read(str(audio_source), dtype="float32")

    # Downmix multi-channel audio to mono
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)

    # Resample to target_sr using polyphase filtering
    if sr != target_sr:
        gcd = np.gcd(int(sr), int(target_sr))
        up = int(target_sr) // gcd
        down = int(sr) // gcd
        signal = scipy.signal.resample_poly(signal, up, down).astype(np.float32)
        sr = target_sr

    return signal.astype(np.float32), sr


def preprocess_audio(
    signal: np.ndarray,
    sr: int = SAMPLE_RATE,
    target_samples: int = TARGET_SAMPLES,
    normalize: bool = True,
) -> np.ndarray:
    """
    Audio preprocessing:
    1. Trims silence using frame-level energy thresholding.
    2. Centers and pads/truncates to exact target_samples (66,150 samples = 3.0s).
    3. Normalizes peak amplitude to [-1.0, 1.0].
    """
    # 1. Energy-based silence trimming
    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)
    if len(signal) > frame_len:
        num_frames = 1 + int((len(signal) - frame_len) / hop)
        if num_frames > 0:
            strided = np.lib.stride_tricks.as_strided(
                signal, shape=(num_frames, frame_len),
                strides=(signal.strides[0] * hop, signal.strides[0])
            )
            frame_energy = np.mean(strided ** 2, axis=1)
            thresh = np.max(frame_energy) * 0.001
            active_idx = np.where(frame_energy > thresh)[0]
            if len(active_idx) > 10:
                start_sample = max(0, active_idx[0] * hop)
                end_sample = min(len(signal), active_idx[-1] * hop + frame_len)
                if (end_sample - start_sample) > 0.5 * sr:
                    signal = signal[start_sample:end_sample]

    # 2. Fix length (center-aligned pad or crop)
    if len(signal) < target_samples:
        pad_total = target_samples - len(signal)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        signal = np.pad(signal, (pad_left, pad_right), mode="constant")
    elif len(signal) > target_samples:
        start = (len(signal) - target_samples) // 2
        signal = signal[start : start + target_samples]

    # 3. Peak normalization
    if normalize:
        peak = np.max(np.abs(signal))
        if peak > 1e-6:
            signal = signal / peak

    return signal.astype(np.float32)


# Data Augmentation Functions
def augment_add_noise(signal: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    """Adds subtle Gaussian background noise."""
    noise = np.random.randn(len(signal)).astype(np.float32)
    return signal + noise_factor * noise


def augment_time_shift(signal: np.ndarray, shift_max_ratio: float = 0.1) -> np.ndarray:
    """Rolls audio temporally with zero padding."""
    shift = int(np.random.uniform(-shift_max_ratio, shift_max_ratio) * len(signal))
    augmented = np.roll(signal, shift)
    if shift > 0:
        augmented[:shift] = 0
    elif shift < 0:
        augmented[shift:] = 0
    return augmented


def augment_pitch_resample(signal: np.ndarray, factor: float = 1.05) -> np.ndarray:
    """Simulates pitch shifting by slight resampling and length fixing."""
    new_len = int(len(signal) * factor)
    resampled = scipy.signal.resample(signal, new_len).astype(np.float32)
    return preprocess_audio(resampled, target_samples=len(signal), normalize=False)


def _compute_mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Constructs triangular Slaney-normalized Mel filterbank matrix."""
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    m_pts = np.linspace(hz_to_mel(0), hz_to_mel(sr / 2), n_mels + 2)
    hz_pts = mel_to_hz(m_pts)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    n_freqs = n_fft // 2 + 1
    fbank = np.zeros((n_mels, n_freqs), dtype=np.float32)

    for i in range(1, n_mels + 1):
        f_left, f_mid, f_right = bin_pts[i - 1], bin_pts[i], bin_pts[i + 1]
        for f in range(f_left, f_mid):
            if f < n_freqs:
                fbank[i - 1, f] = (f - f_left) / max(1, f_mid - f_left)
        for f in range(f_mid, f_right):
            if f < n_freqs:
                fbank[i - 1, f] = (f_right - f) / max(1, f_right - f_mid)

    # Slaney area normalization
    enorm = 2.0 / (hz_pts[2 : n_mels + 2] - hz_pts[:n_mels])
    fbank *= enorm[:, np.newaxis]
    return fbank


# Precomputed Mel Filterbank Matrix for efficiency
_GLOBAL_MEL_FBANK_40 = _compute_mel_filterbank(SAMPLE_RATE, N_FFT, N_MFCC)
_GLOBAL_MEL_FBANK_64 = _compute_mel_filterbank(SAMPLE_RATE, N_FFT, 64)


def extract_acoustic_features(
    signal: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts acoustic feature representations:
    1. mfcc_seq: Shape (126, 40) temporal sequence for 1D CNN / BiLSTM / Hybrid models.
    2. feat_1d: Shape (215,) comprehensive acoustic statistics:
       - 40 MFCC mean + 40 MFCC std (80)
       - 64 Mel-band energy mean + 64 Mel-band energy std (128)
       - Spectral Centroid mean + std (2)
       - RMS Energy mean, max, std (3)
       - Zero-Crossing Rate mean + std (2)
    """
    window = np.hanning(n_fft).astype(np.float32)
    num_frames = 1 + int((len(signal) - n_fft) / hop_length)

    frames = np.lib.stride_tricks.as_strided(
        signal, shape=(num_frames, n_fft),
        strides=(signal.strides[0] * hop_length, signal.strides[0])
    )
    windowed = frames * window
    spec = np.abs(np.fft.rfft(windowed, n=n_fft)) ** 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    # 1. 40-MFCC Sequence
    mel_spec_40 = np.dot(spec, _GLOBAL_MEL_FBANK_40.T)
    log_mel_40 = np.log(np.maximum(mel_spec_40, 1e-10))
    mfcc_seq = scipy.fftpack.dct(log_mel_40, type=2, axis=-1, norm="ortho")[:, :n_mfcc].astype(np.float32)

    # 2. 64 Mel Spectrogram
    mel_spec_64 = np.dot(spec, _GLOBAL_MEL_FBANK_64.T)
    log_mel_64 = np.log(np.maximum(mel_spec_64, 1e-10)).astype(np.float32)

    # 3. Acoustic Statistics for 1D Feature Vector
    mfcc_mean = np.mean(mfcc_seq, axis=0)
    mfcc_std = np.std(mfcc_seq, axis=0)
    mel_mean = np.mean(log_mel_64, axis=0)
    mel_std = np.std(log_mel_64, axis=0)

    # Spectral Centroid
    sc = np.sum(freqs * spec, axis=1) / (np.sum(spec, axis=1) + 1e-10)
    sc_stats = [np.mean(sc), np.std(sc)]

    # RMS Energy
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    rms_stats = [np.mean(rms), np.max(rms), np.std(rms)]

    # Zero Crossing Rate
    zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=1)) > 0, axis=1)
    zcr_stats = [np.mean(zcr), np.std(zcr)]

    feat_1d = np.concatenate([
        mfcc_mean,
        mfcc_std,
        mel_mean,
        mel_std,
        sc_stats,
        rms_stats,
        zcr_stats
    ]).astype(np.float32)

    return mfcc_seq, feat_1d


def process_dataset_features(
    metadata_df: pd.DataFrame,
    augment: bool = True,
    output_dir: Path = PROCESSED_DIR,
) -> Dict[str, Any]:
    """
    Extracts features for all recordings in metadata_df with augmentation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    emotion_to_id = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}
    id_to_emotion = {idx: emotion for idx, emotion in enumerate(EMOTIONS)}

    X_1d_list = []
    X_seq_list = []
    y_str_list = []
    y_id_list = []
    actor_ids = []
    genders = []

    print(f"[Feature Extractor] Processing {len(metadata_df)} audio files (Augment={augment})...")

    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Extracting Features"):
        filepath = row["filepath"]
        emotion = row["emotion"]
        if emotion not in emotion_to_id:
            continue

        try:
            raw_sig, sr = load_audio(filepath, target_sr=SAMPLE_RATE)
            clean_sig = preprocess_audio(raw_sig, sr=SAMPLE_RATE)

            # 1. Original Sample
            seq, f1d = extract_acoustic_features(clean_sig, sr=SAMPLE_RATE)
            X_1d_list.append(f1d)
            X_seq_list.append(seq)
            y_str_list.append(emotion)
            y_id_list.append(emotion_to_id[emotion])
            actor_ids.append(row["actor_id"])
            genders.append(row["gender"])

            # 2. Data Augmentation: Noise Injection
            if augment:
                sig_noise = preprocess_audio(augment_add_noise(clean_sig, noise_factor=0.005))
                seq_n, f1d_n = extract_acoustic_features(sig_noise, sr=SAMPLE_RATE)
                X_1d_list.append(f1d_n)
                X_seq_list.append(seq_n)
                y_str_list.append(emotion)
                y_id_list.append(emotion_to_id[emotion])
                actor_ids.append(row["actor_id"])
                genders.append(row["gender"])

                # 3. Data Augmentation: Time Shift
                sig_shift = preprocess_audio(augment_time_shift(clean_sig, shift_max_ratio=0.08))
                seq_s, f1d_s = extract_acoustic_features(sig_shift, sr=SAMPLE_RATE)
                X_1d_list.append(f1d_s)
                X_seq_list.append(seq_s)
                y_str_list.append(emotion)
                y_id_list.append(emotion_to_id[emotion])
                actor_ids.append(row["actor_id"])
                genders.append(row["gender"])

        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue

    X_1d = np.array(X_1d_list, dtype=np.float32)
    X_seq = np.array(X_seq_list, dtype=np.float32)
    y_str = np.array(y_str_list)
    y_id = np.array(y_id_list, dtype=np.int32)
    actor_ids = np.array(actor_ids, dtype=np.int32)
    genders = np.array(genders)

    print(f"\n[Feature Extractor] Feature extraction complete!")
    print(f"  - X_1d shape: {X_1d.shape} (1D features)")
    print(f"  - X_seq shape: {X_seq.shape} (Temporal sequences)")
    print(f"  - Total samples: {len(y_id)}")

    # Save to disk
    np.save(output_dir / "X_1d.npy", X_1d)
    np.save(output_dir / "X_seq.npy", X_seq)
    np.save(output_dir / "y_str.npy", y_str)
    np.save(output_dir / "y_id.npy", y_id)
    np.save(output_dir / "actor_ids.npy", actor_ids)
    np.save(output_dir / "genders.npy", genders)

    label_meta = {
        "emotion_to_id": emotion_to_id,
        "id_to_emotion": id_to_emotion,
        "emotions": EMOTIONS,
        "sample_rate": SAMPLE_RATE,
        "audio_duration": AUDIO_DURATION,
        "target_samples": TARGET_SAMPLES,
        "target_frames": X_seq.shape[1],
        "n_mfcc": N_MFCC,
        "num_1d_features": X_1d.shape[1],
        "total_samples": len(y_id)
    }
    with open(output_dir / "features_meta.json", "w") as f:
        json.dump(label_meta, f, indent=2)

    print(f"[Feature Extractor] Saved feature artifacts to {output_dir}")
    return label_meta


def prepare_single_audio_for_inference(
    audio_input: Union[str, Path, io.BytesIO, bytes, np.ndarray],
    expected_format: str = "seq",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Production helper for FastAPI backend inference.
    Preprocesses uploaded/recorded audio and returns the input tensor + acoustic metrics.
    """
    raw_sig, sr = load_audio(audio_input, target_sr=SAMPLE_RATE)
    clean_sig = preprocess_audio(raw_sig, sr=SAMPLE_RATE)

    seq, feat_1d = extract_acoustic_features(clean_sig, sr=SAMPLE_RATE)

    # Compute key acoustic diagnostics
    rms = float(np.sqrt(np.mean(clean_sig ** 2)))
    zcr = float(np.mean(np.abs(np.diff(np.sign(clean_sig))) > 0))
    duration = float(len(clean_sig) / SAMPLE_RATE)

    # Approximate spectral centroid
    fft_mag = np.abs(np.fft.rfft(clean_sig[:2048]))
    freqs = np.fft.rfftfreq(2048, d=1.0 / SAMPLE_RATE)
    centroid = float(np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-9))

    diagnostics = {
        "duration_seconds": round(duration, 2),
        "sample_rate": SAMPLE_RATE,
        "mean_rms_energy": round(rms, 4),
        "mean_zero_crossing_rate": round(zcr, 4),
        "spectral_centroid_hz": round(centroid, 1)
    }

    if expected_format == "1d":
        tensor = np.expand_dims(feat_1d, axis=0)  # (1, 215)
    else:
        tensor = np.expand_dims(seq, axis=0)      # (1, 126, 40)

    return tensor, diagnostics


if __name__ == "__main__":
    metadata_path = PROCESSED_DIR / "ravdess_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {metadata_path}. Run dataset_loader first.")
    df = pd.read_csv(metadata_path)
    process_dataset_features(df, augment=True)
