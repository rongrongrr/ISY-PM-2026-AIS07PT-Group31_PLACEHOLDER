# Training

This directory is where the compiled CREMA-D dataset lives locally and where team members create and store their model training notebooks.

---

## Critical: Do not push the `data/` folder to GitHub

> **The `data/` directory must never be committed or pushed.**

It contains ~14,884 files (7,442 WAV + 7,442 PNG) totalling several gigabytes. It is listed in `.gitignore` and will be ignored by git automatically. Do not force-add it.

---

## Getting the dataset

The `data/` folder is not included in the repository. Download it from the shared drive:

**[Download data.zip](https://drive.google.com/file/d/1G9g2tCyTPHySN-g7q3OT08lfUKJ9MeiT/view?usp=share_link)**

> Note: this is a temporary team link while the project is in development. A permanent public link will be made available upon submission.

Once downloaded, extract it so that the directory structure looks like this:

```
SystemCode/data/training/
├── data/
│   ├── train/
│   │   ├── ANG/
│   │   ├── DIS/
│   │   ├── FEA/
│   │   ├── HAP/
│   │   ├── NEU/
│   │   └── SAD/
│   └── test/
│       ├── ANG/
│       ├── DIS/
│       ├── FEA/
│       ├── HAP/
│       ├── NEU/
│       └── SAD/
├── training_template.ipynb
└── README.md   ← you are here
```

Each emotion subdirectory contains paired `.wav` and `.png` files (mel-spectrogram). Use the PNGs for image-based models (CNN, ViT, YOLO, DETR) and the WAVs for audio-based models.

---

## Creating your training notebook

Each team member is expected to create their own `.ipynb` training notebook in this directory. Name it something descriptive, e.g. `train_cnn_baseline.ipynb`, `train_resnet50.ipynb`, `train_yolo.ipynb`.

A template is provided to get you started: **`training_template.ipynb`**

The template is self-contained — it installs its own Python packages via `!pip install` cells, following the same pattern as the prep notebooks. You only need to have a Python environment active before launching Jupyter.

**Environment setup (pick one):**

```bash
# venv
python -m venv train-env
source train-env/bin/activate   # Windows: train-env\Scripts\activate
pip install jupyter
jupyter notebook

# conda / miniconda
conda create -n train-env python=3.10
conda activate train-env
conda install jupyter
jupyter notebook
```

Open `training_template.ipynb` from this directory and run cells top-to-bottom. Adjust the configuration cell (batch size, learning rate, image size, number of epochs) and swap in your own model architecture where indicated.
