from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import os

try:
    import pandas as pd
except Exception:
    pd = None  # Only needed if you call the CSV readers

ENV_VAR = "FSD50K_DIR"

def _candidate_roots() -> List[Path]:
    cands: List[Path] = []
    env_path = os.getenv(ENV_VAR)
    if env_path:
        cands.append(Path(env_path).expanduser().resolve())

    repo_root = Path(__file__).resolve().parents[1]
    cands.append(repo_root / "data" / "raw" / "FSD50K")

    cands.append(Path("/content/drive/MyDrive/FSD50K"))
    cands.append(Path("/content/drive/MyDrive/Data/FSD50K"))

    seen = set(); out = []
    for p in cands:
        if str(p) not in seen:
            out.append(p); seen.add(str(p))
    return out

def find_fsd50k_root(strict: bool = False) -> Optional[Path]:
    required = {"FSD50K.dev_audio", "FSD50K.eval_audio", "FSD50K.ground_truth", "FSD50K.metadata"}
    for cand in _candidate_roots():
        if cand.exists() and cand.is_dir():
            names = {child.name for child in cand.iterdir()}
            if required.issubset(names):
                return cand
    if strict:
        raise FileNotFoundError(
            "Could not locate FSD50K. "
            f"Tried: {', '.join(str(p) for p in _candidate_roots())}. "
            f"Set {ENV_VAR} or place it under data/raw/FSD50K/"
        )
    return None

def list_wavs(split: str = "dev") -> List[Path]:
    root = find_fsd50k_root(strict=True)
    if split not in {"dev", "eval"}:
        raise ValueError("split must be 'dev' or 'eval'")
    folder = root / f"FSD50K.{split}_audio"
    return sorted(folder.glob("*.wav"))

def load_ground_truth(split: str = "dev"):
    if pd is None:
        raise ImportError("pandas is required for load_ground_truth(...)")
    root = find_fsd50k_root(strict=True)
    gt = root / "FSD50K.ground_truth"
    csv_path = gt / ("dev.csv" if split == "dev" else "eval.csv")
    return pd.read_csv(csv_path)

def load_vocabulary():
    if pd is None:
        raise ImportError("pandas is required for load_vocabulary(...)")
    root = find_fsd50k_root(strict=True)
    return pd.read_csv(root / "FSD50K.ground_truth" / "vocabulary.csv")

def print_dataset_summary() -> None:
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
