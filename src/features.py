"""
Audio feature helpers for the FSD50K project.

This module keeps feature-extraction logic small and dependency-light.
It **lazy-imports `librosa`** only when a function that needs it is called,
so importing `src.features` won’t fail on machines without librosa installed.

Provided utilities
------------------
- `load_wav_mono(path, sr=32000)` → (y, sr)
    Load an audio file as **mono float32**, optionally resampling to `sr`.

- `logmel(y, sr, n_mels=64)` → (n_mels, time)
    Compute **log-mel spectrogram** in dB (librosa’s `power_to_db` with `ref=np.max`).

- `mfcc(y, sr, n_mfcc=20)` → (n_mfcc, time)
    Compute **MFCC** coefficients.

- `frame_feature(mat, max_frames=None)` → (n_features, time’)
    Clip/pad features along the **time** axis to a fixed number of frames.

- `save_numpy(array, out_path)`:
    Save a NumPy array to disk, creating parent directories if needed.

Quick example
-------------
>>> from pathlib import Path
>>> from src.features import load_wav_mono, logmel, frame_feature, save_numpy
>>> y, sr = load_wav_mono(Path("data/raw/FSD50K.dev_audio/63.wav"), sr=32000)
>>> X = logmel(y, sr, n_mels=64)           # (64, T)
>>> X_fixed = frame_feature(X, max_frames=512)
>>> save_numpy(X_fixed, Path("data/processed/logmel/63.npy"))

Notes
-----
- Shapes follow the common convention **(n_features, time)**.
- If you change `sr` during loading, librosa will resample internally.
- Install librosa only if you plan to **execute** these functions:
    pip install librosa
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional
import numpy as np


# --- Lazy dependency handling -------------------------------------------------
def _lazy_import_librosa():
    """
    Import `librosa` only when needed.

    This keeps module import cheap and avoids hard dependency errors when someone
    just imports `src.features` without running audio code.
    """
    import importlib
    return importlib.import_module("librosa")


# --- I/O ----------------------------------------------------------------------
def load_wav_mono(path: Path, sr: int = 32000) -> Tuple[np.ndarray, int]:
    """
    Load an audio file as **mono** float32 and (optionally) resample.

    Parameters
    ----------
    path : Path
        Path to an audio file readable by librosa (e.g., WAV/MP3/OGG).
    sr : int, default=32000
        Target sample rate. If None, the file’s native sr is preserved.

    Returns
    -------
    y : np.ndarray, shape (samples,)
        Mono waveform in float32, amplitude range roughly [-1, 1].
    sr : int
        Actual sample rate of `y` (equals the `sr` argument unless sr=None).
    """
    librosa = _lazy_import_librosa()
    y, sr = librosa.load(str(path), sr=sr, mono=True)
    return y, sr


# --- Features -----------------------------------------------------------------
def logmel(y: np.ndarray, sr: int, n_mels: int = 64) -> np.ndarray:
    """
    Compute **log-mel spectrogram** (in dB).

    Parameters
    ----------
    y : np.ndarray, shape (samples,)
        Mono audio waveform.
    sr : int
        Sample rate of `y`.
    n_mels : int, default=64
        Number of mel bins (frequency channels).

    Returns
    -------
    logS : np.ndarray, shape (n_mels, time)
        Log-scaled mel spectrogram (dB). Uses `ref=np.max` so values are <= 0.
    """
    librosa = _lazy_import_librosa()
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)  # power mel spec
    return librosa.power_to_db(S, ref=np.max)


def mfcc(y: np.ndarray, sr: int, n_mfcc: int = 20) -> np.ndarray:
    """
    Compute **MFCC** coefficients.

    Parameters
    ----------
    y : np.ndarray, shape (samples,)
        Mono audio waveform.
    sr : int
        Sample rate of `y`.
    n_mfcc : int, default=20
        Number of MFCC coefficients to keep.

    Returns
    -------
    mfcc_mat : np.ndarray, shape (n_mfcc, time)
        MFCC feature matrix.
    """
    librosa = _lazy_import_librosa()
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)


# --- Utilities ----------------------------------------------------------------
def frame_feature(mat: np.ndarray, max_frames: Optional[int] = None) -> np.ndarray:
    """
    Optionally clip/pad a feature matrix along the **time** axis.

    This is handy for batching models that expect a fixed number of frames.

    Parameters
    ----------
    mat : np.ndarray, shape (n_features, time)
        Feature matrix (e.g., log-mel or MFCC).
    max_frames : int | None
        If None, return `mat` unchanged.
        If int, output will have time dimension == `max_frames`:
          - If `time` > `max_frames`, it is **truncated**.
          - If `time` < `max_frames`, it is **zero-padded** on the right.

    Returns
    -------
    out : np.ndarray, shape (n_features, max_frames) or (n_features, time)
    """
    if max_frames is None:
        return mat
    n_feat, T = mat.shape
    out = np.zeros((n_feat, max_frames), dtype=mat.dtype)
    T_use = min(T, max_frames)
    out[:, :T_use] = mat[:, :T_use]
    return out


def save_numpy(array: np.ndarray, out_path: Path) -> None:
    """
    Save a NumPy array to `out_path` (parent directories created if missing).

    Parameters
    ----------
    array : np.ndarray
        Array to save.
    out_path : Path
        Destination path ending with `.npy`.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, array)
