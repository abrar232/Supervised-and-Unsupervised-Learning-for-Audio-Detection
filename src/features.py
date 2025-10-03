from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import numpy as np

# librosa is only needed when you *actually* run these functions
def _lazy_import_librosa():
    import importlib
    return importlib.import_module("librosa")

def load_wav_mono(path: Path, sr: int = 32000) -> Tuple[np.ndarray, int]:
    """Load audio as mono float32. (Requires librosa when executed.)"""
    librosa = _lazy_import_librosa()
    y, sr = librosa.load(str(path), sr=sr, mono=True)
    return y, sr

def logmel(y: np.ndarray, sr: int, n_mels: int = 64) -> np.ndarray:
    """Log-mel spectrogram in dB (requires librosa when executed)."""
    librosa = _lazy_import_librosa()
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    return librosa.power_to_db(S, ref=np.max)

def mfcc(y: np.ndarray, sr: int, n_mfcc: int = 20) -> np.ndarray:
    """MFCC features."""
    librosa = _lazy_import_librosa()
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

def frame_feature(mat: np.ndarray, max_frames: Optional[int] = None) -> np.ndarray:
    """
    Optionally clip/pad time frames to a fixed length for batching.
    Expects (n_features, time). Pads with zeros if shorter.
    """
    if max_frames is None:
        return mat
    n_feat, T = mat.shape
    out = np.zeros((n_feat, max_frames), dtype=mat.dtype)
    T_use = min(T, max_frames)
    out[:, :T_use] = mat[:, :T_use]
    return out

def save_numpy(array: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, array)
