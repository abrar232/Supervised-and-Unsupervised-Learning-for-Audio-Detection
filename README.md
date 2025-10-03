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
│  ├─ __init__.py
│  ├─ utils.py               # paths, seeding
│  ├─ dataio.py              # file listing / simple split helpers
│  └─ features.py            # audio features (e.g., log-mel, MFCC)
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

## Results (fill these in with your actual metrics)
Approach	Features	Model / Method	Metric (e.g., F1/AUC)	Notes
Supervised	Log-mel	CNN / [framework]		
Unsupervised	MFCC + k-means	k-means (k=…)		
Unsupervised	Log-mel	Isolation Forest		

## Outputs & Models
Trained weights are saved to models/ (ignored by Git).

Figures/logs/metrics go to outputs/ (ignored).

Keep the repo small and fast by not committing large artifacts.

## Tips
In Google Colab: Edit ▸ Clear all outputs before saving the notebook.

Locally, consider nbstripout to automatically remove outputs on commit:

pip install nbstripout
nbstripout --install
