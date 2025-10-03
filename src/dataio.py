# src/dataio.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import os

try:
    import pandas as pd
except Exception:
    pd = None  # Only needed if the CSV readers are called


# ---------- Configuration ----------

# Dataset location can be overriden by setting an env var:
#   FSD50K_DIR=/absolute/path/to/FSD50K
ENV_VAR = "FSD50K_DIR"


# ---------- Path resolution ----------

def _candidate_roots() -> List[Path]:
    """
    Ordered list of places to look for the FSD50K folder.
    1) ENV var FSD50K_DIR (if set)
    2) repo-local: data/raw/FSD50K
    3) common Colab mounts
    """
    cands: List[Path] = []

    # 1) Environment variable
    env_path = os.getenv(ENV_VAR)
    if env_path:
        cands.append(Path(env_path).expanduser().resolve())

    # 2) Repo local (assuming this file lives in <repo>/src/dataio.py)
    repo_root = Path(__file__).resolve().parents[1]
    cands.append(repo_root / "data" / "raw" / "FSD50K")

    # 3) Common Google Colab mount locations (adjust if yours is different)
    cands.append(Path("/content/drive/MyDrive/FSD50K"))
    cands.append(Path("/content/drive/MyDrive/Data/FSD50K"))

    # Remove duplicates while preserving order
    seen = set()
    out = []
    for p in cands:
        if str(p) not in seen:
            out.append(p)
            seen.add(str(p))
    return out


def find_fsd50k_root(strict: bool = False) -> Optional[Path]:
    """
    Return the first existing path that looks like an FSD50K root.
    Expected subfolders: FSD50K.dev_audio, FSD50K.eval_audio, FSD50K.ground_truth, FSD50K.metadata
    """
    required = {"FSD50K.dev_audio", "FSD50K.eval_audio", "FSD50K.ground_truth", "FSD50K.metadata"}

    for cand in _candidate_roots():
        if cand.exists() and cand.is_dir():
            names = {p.name for p in cand.iterdir() if p.is_dir() or p.is_file()}
            if required.issubset(names):
                return cand

    if strict:
        raise FileNotFoundError(
            "Could not locate FSD50K. "
            f"Tried: {', '.join(str(p) for p in _candidate_roots())}. "
            f"Set {ENV_VAR} or place the dataset under data/raw/FSD50K/"
        )
    return None


# ---------- Listing audio ----------

def list_wavs(split: str = "dev") -> List[Path]:
    """
    List WAV files for a given split: 'dev' or 'eval'.
    """
    root = find_fsd50k_root(strict=True)
    if split not in {"dev", "eval"}:
        raise ValueError("split must be 'dev' or 'eval'")

    folder = root / (f"FSD50K.{split}_audio")
    return sorted(folder.glob("*.wav"))


# ---------- Ground-truth & metadata ----------

def load_ground_truth(split: str = "dev") -> "pd.DataFrame":
    """
    Load ground-truth CSVs for 'dev' or 'eval'.
    Requires pandas. Columns come from official FSD50K CSVs.
    """
    if pd is None:
        raise ImportError("pandas is required for load_ground_truth(...)")

    root = find_fsd50k_root(strict=True)
    gt = root / "FSD50K.ground_truth"
    if split == "dev":
        csv_path = gt / "dev.csv"
    elif split == "eval":
        csv_path = gt / "eval.csv"
    else:
        raise ValueError("split must be 'dev' or 'eval'")

    return pd.read_csv(csv_path)


def load_vocabulary() -> "pd.DataFrame":
    """
    Load the vocabulary.csv mapping (label id/name).
    """
    if pd is None:
        raise ImportError("pandas is required for load_vocabulary(...)")

    root = find_fsd50k_root(strict=True)
    csv_path = root / "FSD50K.ground_truth" / "vocabulary.csv"
    return pd.read_csv(csv_path)


def load_metadata_file(name: str = "dev_clips_info_FSD50K.json") -> Path:
    """
    Return the path to a file inside FSD50K.metadata.
    Common names:
      - 'dev_clips_info_FSD50K.json'
      - 'eval_clips_info_FSD50K.json'
      - 'class_info_FSD50K.json'
      - 'pp_pnp_ratings_FSD50K.json'
    """
    root = find_fsd50k_root(strict=True)
    path = root / "FSD50K.metadata" / name
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    return path


# ---------- Helpers for thw notebook ----------

def print_dataset_summary() -> None:
    """
    Quick sanity check to print where the dataset is and counts of WAVs.
    """
    root = find_fsd50k_root(strict=False)
    if root is None:
        print("FSD50K not found. Set FSD50K_DIR or place dataset under data/raw/FSD50K/")
        for i, c in enumerate(_candidate_roots(), 1):
            print(f"  candidate {i}: {c}")
        return

    dev_n = len(list_wavs("dev"))
    eval_n = len(list_wavs("eval"))
    print(f"FSD50K root: {root}")
    print(f"dev wavs:  {dev_n}")
    print(f"eval wavs: {eval_n}")
