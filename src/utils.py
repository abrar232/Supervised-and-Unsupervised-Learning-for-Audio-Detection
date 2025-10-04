"""
Project-wide utilities: paths, seeding, tiny JSON helpers, and timestamps.

- `project_paths()` returns commonly used directories based on the assumption
  that this file lives in `<repo>/src/utils.py`. If you move it, adjust the
  `parents[1]` logic accordingly.

- `set_seed()` seeds Python, NumPy, and (if available) PyTorch/TensorFlow so
  runs are more reproducible.

- `ensure_dirs()` safely creates directories.

- `save_json()` / `load_json()` are tiny wrappers around JSON I/O.

- `timestamp()` generates a filesystem-friendly time string you can use
  for experiment/run folder names.
"""

from __future__ import annotations
from pathlib import Path
import random, json, numpy as np
from datetime import datetime


def project_paths() -> dict[str, Path]:
    """
    Build a dictionary of common project paths relative to the **repo root**.

    Assumes this file is at: <repo>/src/utils.py  → repo root = parents[1]

    Returns
    -------
    dict[str, Path]
        Keys: "root", "data", "raw", "processed", "outputs", "models",
        "notebooks", "src".
    """
    root = Path(__file__).resolve().parents[1]
    return {
        "root": root,
        "data": root / "data",
        "raw": root / "data" / "raw",
        "processed": root / "data" / "processed",
        "outputs": root / "outputs",
        "models": root / "models",
        "notebooks": root / "notebooks",
        "src": root / "src",
    }


def set_seed(seed: int = 42) -> None:
    """
    Seed Python/NumPy and, if present, deep-learning libs for reproducibility.

    Parameters
    ----------
    seed : int, default=42
        Seed value used across RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Optional: PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Optionally make CuDNN deterministic (slower but reproducible):
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    # Optional: TensorFlow
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


def ensure_dirs(*paths: Path) -> None:
    """
    Create directories if they don't exist (parents included).

    Example
    -------
    >>> from pathlib import Path
    >>> ensure_dirs(Path("outputs/figures"), Path("outputs/metrics"))
    """
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path) -> None:
    """
    Save a Python object as pretty-printed JSON.

    Parameters
    ----------
    obj : Any
        JSON-serializable object (dict/list/str/num/etc.).
    path : Path
        Destination path; parent directories are created if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def load_json(path: Path):
    """
    Load a JSON file and return the parsed Python object.

    Parameters
    ----------
    path : Path
        Path to a JSON file.

    Returns
    -------
    Any
        Parsed JSON content (dict/list/...).
    """
    return json.loads(Path(path).read_text())


def timestamp(prefix: str = "exp") -> str:
    """
    Generate a filesystem-friendly timestamp string.

    Parameters
    ----------
    prefix : str, default="exp"
        A short prefix to put in front of the timestamp.

    Returns
    -------
    str
        e.g., "exp_2025-10-04_12-34-56"
    """
    return f"{prefix}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
