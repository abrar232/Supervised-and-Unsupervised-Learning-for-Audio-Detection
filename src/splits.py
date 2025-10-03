from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
from src.utils import save_json, ensure_dirs
import random

def random_split(paths: List[Path], val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[Path], List[Path]]:
    rnd = random.Random(seed)
    items = list(paths)
    rnd.shuffle(items)
    n_val = max(1, int(len(items) * val_ratio))
    return items[n_val:], items[:n_val]

def write_split_files(train: List[Path], val: List[Path], out_dir: Path) -> None:
    ensure_dirs(out_dir)
    save_json([str(p) for p in train], out_dir / "train_files.json")
    save_json([str(p) for p in val],   out_dir / "val_files.json")
