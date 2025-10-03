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
.
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ notebooks/
│ └─ Thesis_Code.ipynb # main analysis notebook (no outputs saved)
├─ src/
│ ├─ init.py
│ ├─ utils.py # paths, seeding
│ ├─ dataio.py # file listing / simple split helpers
│ └─ features.py # audio features (e.g., log-mel, MFCC)
├─ data/
│ ├─ raw/ # place original audio here (not committed)
│ └─ processed/ # derived features (not committed)
├─ models/ # trained weights/checkpoints (not committed)
└─ outputs/ # figures, logs, metrics (not committed)

## Setup
- Python 3.10+ recommended.  
- (Optional) Create and activate a virtual environment, then install dependencies:

python -m venv .venv
### Windows: .venv\Scripts\activate
### macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt

## Data
Put your audio files under data/raw/.

If supervised (labeled) classification, use class folders:

data/raw/
  class_a/*.wav
  class_b/*.wav
  ...
If unsupervised/anomaly, you can place WAVs directly under data/raw/ (or any folder structure you prefer). Adjust your notebook paths accordingly.

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
