"""
Deterministic train/val split helpers.

This module creates **reproducible** file lists for training/validation and
(optionally) writes them to disk as small JSON files. It does not inspect
file contents—only shuffles the provided list of Paths.

Provided functions
------------------
- random_split(paths, val_ratio=0.2, seed=42) -> (train_paths, val_paths)
    Seeded shuffle using a dedicated RNG so results are stable.

- write_split_files(train, val, out_dir)
    Save JSON lists of stringified paths: `train_files.json`, `val_files.json`.

Example
-------
>>> from pathlib import Path
>>> files = sorted(Path("data/raw/FSD50K.dev_audio").glob("*.wav"))
>>> train, val = random_split(files, val_ratio=0.2, seed=123)
>>> write_split_files(train, val, Path("data/processed/splits"))
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
from src.utils import save_json, ensure_dirs
import random


def random_split(paths: List[Path], val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[Path], List[Path]]:
    """
    Create a **reproducible** train/val split from a list of file paths.

    The shuffle uses a *local* Random instance seeded with `seed` so the result
    does not depend on global RNG state or other libraries.

    Parameters
    ----------
    paths : list[pathlib.Path]
        Collection of file paths to split.
    val_ratio : float, default=0.2
        Fraction of items assigned to the validation set.
    seed : int, default=42
        Seed for the local RNG.

    Returns
    -------
    train : list[pathlib.Path]
        Training file paths.
    val : list[pathlib.Path]
        Validation file paths.

    Notes
    -----
    - Ensures at least **one** item in the validation set when `paths` is non-empty.
    - For very small datasets, this may leave `train` empty if `len(paths) == 1`.
    """
    rnd = random.Random(seed)   # local RNG → deterministic, no side effects
    items = list(paths)         # do not mutate the caller's list
    rnd.shuffle(items)
    n_val = max(1, int(len(items) * val_ratio)) if len(items) > 0 else 0
    return items[n_val:], items[:n_val]


def write_split_files(train: List[Path], val: List[Path], out_dir: Path) -> None:
    """
    Write train/val path lists as JSON files in `out_dir`.

    Files written:
      - `train_files.json` — list[str]
      - `val_files.json`   — list[str]

    Parameters
    ----------
    train : list[pathlib.Path]
        Training file paths.
    val : list[pathlib.Path]
        Validation file paths.
    out_dir : pathlib.Path
        Destination directory (created if missing).
    """
    ensure_dirs(out_dir)
    # JSON doesn't have a Path type → stringify for portability
    save_json([str(p) for p in train], out_dir / "train_files.json")
    save_json([str(p) for p in val],   out_dir / "val_files.json")
