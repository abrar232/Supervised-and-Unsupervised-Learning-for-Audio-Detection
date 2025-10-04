"""
FSD50K dataset helpers (path discovery + tiny loaders).

This module finds the FSD50K root folder, so the notebooks don’t need hard-coded 
paths. It supports both local usage and Google Colab (Drive-mounted) workflows.

Search order (first match wins):
1) Environment variable `FSD50K_DIR`
2) Repo-local: `<repo>/data/raw/FSD50K`
3) Common Colab Drive paths:
   - `/content/drive/MyDrive/FSD50K`
   - `/content/drive/MyDrive/Data/FSD50K`

Main functions
--------------
- `find_fsd50k_root(strict=False)` → Path | None  
- `list_wavs(split="dev")` → list[Path]
- `load_ground_truth(split="dev")` → pd.DataFrame
- `load_vocabulary()` → pd.DataFrame
- `print_dataset_summary()` → prints where the dataset was found and file counts

Quick usage
-----------
>>> from src.dataio import print_dataset_summary, list_wavs
>>> print_dataset_summary()
FSD50K root: /content/drive/MyDrive/FSD50K
dev wavs:  20430
eval wavs: 10000
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import os

try:
    import pandas as pd
except Exception:
    pd = None  # Only needed if you call the CSV readers

# Name of the environment variable that can explicitly point to FSD50K
ENV_VAR = "FSD50K_DIR"


def _candidate_roots() -> List[Path]:
    """
    Build an **ordered** list of places to look for the FSD50K root.
    The first existing path that contains the expected subfolders wins.
    """
    cands: List[Path] = []

    # 1) Environment variable override
    env_path = os.getenv(ENV_VAR)
    if env_path:
        cands.append(Path(env_path).expanduser().resolve())

    # 2) Repo-local (assumes this file is in <repo>/src/dataio.py)
    repo_root = Path(__file__).resolve().parents[1]
    cands.append(repo_root / "data" / "raw" / "FSD50K")

    # 3) Common Google Colab Drive mounts
    cands.append(Path("/content/drive/MyDrive/FSD50K"))
    cands.append(Path("/content/drive/MyDrive/Data/FSD50K"))

    # Deduplicate while preserving order
    seen = set()
    out: List[Path] = []
    for p in cands:
        if str(p) not in seen:
            out.append(p)
            seen.add(str(p))
    return out


def find_fsd50k_root(strict: bool = False) -> Optional[Path]:
    """
    Try to locate the FSD50K dataset root by checking the candidate locations.

    A valid root must contain these entries:
    - FSD50K.dev_audio
    - FSD50K.eval_audio
    - FSD50K.ground_truth
    - FSD50K.metadata

    Parameters
    ----------
    strict : bool
        If True, raise FileNotFoundError when nothing is found.
        If False, return None when not found.

    Returns
    -------
    Path | None
        Path to the dataset root, or None if not found and strict=False.
    """
    required = {
        "FSD50K.dev_audio",
        "FSD50K.eval_audio",
        "FSD50K.ground_truth",
        "FSD50K.metadata",
    }

    for cand in _candidate_roots():
        if cand.exists() and cand.is_dir():
            names = {child.name for child in cand.iterdir()}
            if required.issubset(names):
                return cand

    if strict:
        raise FileNotFoundError(
            "Could not locate FSD50K. "
            f"Tried: {', '.join(str(p) for p in _candidate_roots())}. "
            f"Set {ENV_VAR} or place it under data/raw/FSD50K/."
        )
    return None


def list_wavs(split: str = "dev") -> List[Path]:
    """
    List all WAV files for a given split: 'dev' or 'eval'.

    Returns
    -------
    list[Path]
        Sorted list of file paths to *.wav files.
    """
    root = find_fsd50k_root(strict=True)
    if split not in {"dev", "eval"}:
        raise ValueError("split must be 'dev' or 'eval'")
    folder = root / f"FSD50K.{split}_audio"
    return sorted(folder.glob("*.wav"))


def load_ground_truth(split: str = "dev"):
    """
    Load the ground-truth CSV for the specified split ('dev' or 'eval').

    Requires `pandas`. The returned DataFrame matches the official CSV columns.
    """
    if pd is None:
        raise ImportError("pandas is required for load_ground_truth(...)")
    root = find_fsd50k_root(strict=True)
    gt = root / "FSD50K.ground_truth"
    csv_path = gt / ("dev.csv" if split == "dev" else "eval.csv")
    return pd.read_csv(csv_path)


def load_vocabulary():
    """
    Load `vocabulary.csv` from `FSD50K.ground_truth`.

    Requires `pandas`. The file maps label ids/names as provided by the dataset.
    """
    if pd is None:
        raise ImportError("pandas is required for load_vocabulary(...)")
    root = find_fsd50k_root(strict=True)
    return pd.read_csv(root / "FSD50K.ground_truth" / "vocabulary.csv")


def print_dataset_summary() -> None:
    """
    Print a quick summary showing where FSD50K was found and how many WAVs exist
    for the 'dev' and 'eval' splits. If not found, print all candidate locations.
    """
    root = find_fsd50k_root(strict=False)
    if root is None:
        print("FSD50K not found. Set FSD50K_DIR or place at data/raw/FSD50K/")
        for i, c in enumerate(_candidate_roots(), 1):
            print(f"  candidate {i}: {c}")
        return

    dev_n = len(list_wavs("dev"))
    eval_n = len(list_wavs("eval"))
    print(f"FSD50K root: {root}")
    print(f"dev wavs:  {dev_n}")
    print(f"eval wavs: {eval_n}")
