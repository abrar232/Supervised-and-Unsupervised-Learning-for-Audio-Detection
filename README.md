# Supervised & Unsupervised Learning for Audio Event Detection

This project compares **supervised** (labeled) and **unsupervised** (clustering/anomaly) approaches for detecting audio events. The main work lives in a single, cleaned notebook (no stored outputs) so the repo stays light and easy to navigate.

## Highlights
- **Task:** Audio event detection / classification / anomaly detection  
- **Approaches:**  
  - *Supervised:* CNN-based pipeline (e.g., PyTorch/TensorFlow)  
  - *Unsupervised:* Feature extraction + clustering or anomaly methods (e.g., k-means, Isolation Forest)  
- **Notebook:** `notebooks/Thesis_Code.ipynb` (outputs cleared)  
- **Status:** Structured as a portfolio-ready repo; data and large artifacts are gitignored

## Project Structure
```text
.
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ notebooks/
│  └─ Thesis_Code.ipynb      # main analysis notebook (no outputs saved)
├─ src/
    ├─ __init__.py            # makes "src" importable as a package
    ├─ utils.py               # paths, seeding, tiny JSON helpers (save/load), timestamps
    ├─ dataio.py              # finds FSD50K (local or Colab), lists WAVs, loads CSV/metadata
    ├─ features.py            # audio I/O + features (log-mel, MFCC), simple framing & save
    └─ splits.py              # deterministic train/val split + writers
├─ data/
    ├─ raw/
    │  └─ FSD50K/
    │      ├─ FSD50K.dev_audio/        # WAV files for training
    │      ├─ FSD50K.eval_audio/       # WAV files for evaluation
    │      ├─ FSD50K.ground_truth/     # CSV label files
    │      └─ FSD50K.metadata/         # JSON metadata    
    └─ processed/                      # any derived features you generate
├─ models/                   # trained weights/checkpoints (not committed)
└─ outputs/                  # figures, logs, metrics (not committed)

```

## Setup
- Python 3.10+ recommended.  
- (Optional) Create and activate a virtual environment, then install dependencies:

python -m venv .venv

Windows: .venv\Scripts\activate

macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt

### Quick reference (what to import)

python

from src.utils import project_paths, set_seed

from src.dataio import find_fsd50k_root, list_wavs, load_ground_truth, load_vocabulary

from src.features import load_wav_mono, logmel, mfcc, frame_feature

from src.splits import random_split, write_split_files

## Data

This project uses the [FSD50K dataset](https://zenodo.org/record/4060432).  

To reproduce results:

1. Download the dataset from the link above (requires Zenodo account).
2. Place the extracted folders under `data/raw/FSD50K/` like so:
   
data/raw/FSD50K/

FSD50K.dev_audio/

FSD50K.eval_audio/

FSD50K.ground_truth/

FSD50K.metadata/

FSD50K.doc/

Do not commit datasets to GitHub. Keep them locally (they’re ignored by .gitignore).

## How to Use
Open notebooks/Thesis_Code.ipynb.

Run cells as needed to reproduce feature extraction, training, and evaluation.

If you just want to browse the code or share the repo, you don’t need to run anything. The notebook is kept without outputs so Git diffs stay small.

## Outputs 

### Unsupervised embeddings (t-SNE)
![t-SNE scatter of features](outputs/figures/tsne.png)

### ROC curve (example class)
![ROC curve](outputs/figures/roc_curve.png)	

# Models Folder

This folder is for trained model checkpoints and related metadata.  
Model weights are **not committed to GitHub** (they are too large).  

Typical contents after training:
- `checkpoints/` — multiple saved models during training
- `best.pth` or `best.keras` — best model checkpoint
- `config.json` — hyperparameters and training settings
- `metrics.json` — evaluation metrics
- `model_card.md` — short description of the experiment

## Tips
In Google Colab: Edit ▸ Clear all outputs before saving the notebook.

Locally, consider nbstripout to automatically remove outputs on commit:

pip install nbstripout
nbstripout --install
