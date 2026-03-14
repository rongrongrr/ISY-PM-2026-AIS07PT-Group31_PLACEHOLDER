# Data Preparation

This directory contains three Jupyter notebooks that reproduce the full CREMA-D data pipeline from scratch. They are committed to this repository for **posterity, accountability, and reproducibility** — so that the exact steps used to produce the dataset can be inspected, verified, and re-run by anyone.

---

## Critical: Run these notebooks OUTSIDE this repository

> **Do not run these notebooks from inside this project directory.**

The setup notebook clones the CREMA-D mirror repository (several GB via Git LFS) into whichever directory it is run from. If you run it from inside this project, the cloned repo will end up nested inside our own — do not do this.

**Recommended workflow:**

1. Create a working directory somewhere outside this project, e.g.:
   ```
   ~/crema-d-workdir/
   ```
2. Copy or symlink the three notebooks into that directory, or simply note the paths and open them from there in Jupyter.
3. Run the notebooks from that external directory.
4. Once complete, move the output `data/` folder into `SystemCode/data/training/`:
   ```
   mv ~/crema-d-workdir/data /path/to/this/repo/SystemCode/data/training/data
   ```

---

## Environment setup

The notebooks install their own Python packages via `!pip install` cells — you do not need a project-specific environment. However, you **must** have a Python environment active before launching Jupyter. Use whichever tool you prefer:

**venv:**
```bash
python -m venv crema-env
source crema-env/bin/activate   # Windows: crema-env\Scripts\activate
pip install jupyter
jupyter notebook
```

**conda / miniconda:**
```bash
conda create -n crema-env python=3.10
conda activate crema-env
conda install jupyter
jupyter notebook
```

Once Jupyter is running, open each notebook from your external working directory (not from inside this repo).

---

## Notebook execution order

Run the notebooks **in this order**. Each one depends on the output of the previous.

| Step | Notebook | What it does |
|------|----------|--------------|
| 1 | `crema_d_setup.ipynb` | Clones the CREMA-D mirror repo and pulls all audio/video files via Git LFS (~several GB). |
| 2 | `crema_d_prep.ipynb` | Splits the 7,442 WAV files into actor-disjoint train/test sets (80/20 by actor) and organises them into `data/train/{emotion}/` and `data/test/{emotion}/`. |
| 3 | `crema_d_spectrograms.ipynb` | Trims silence from each WAV and generates a grayscale 640×640 mel-spectrogram PNG alongside it. |

After all three notebooks complete successfully, the output is a `data/` directory containing:

```
data/
├── train/
│   ├── ANG/   (1006 WAV + 1006 PNG)
│   ├── DIS/   (1006 WAV + 1006 PNG)
│   ├── FEA/   (1006 WAV + 1006 PNG)
│   ├── HAP/   (1006 WAV + 1006 PNG)
│   ├── NEU/    (860 WAV +  860 PNG)
│   └── SAD/   (1006 WAV + 1006 PNG)
└── test/
    ├── ANG/    (265 WAV +  265 PNG)
    ├── DIS/    (265 WAV +  265 PNG)
    ├── FEA/    (265 WAV +  265 PNG)
    ├── HAP/    (265 WAV +  265 PNG)
    ├── NEU/    (227 WAV +  227 PNG)
    └── SAD/    (265 WAV +  265 PNG)
```

Move this `data/` folder into `SystemCode/data/training/` before starting any model training.
