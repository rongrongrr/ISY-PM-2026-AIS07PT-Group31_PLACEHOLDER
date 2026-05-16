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

### Filename format

Each file is named `{ActorID}_{SentenceCode}_{EmotionCode}_{LevelCode}`, e.g. `1001_IEO_HAP_HI.wav`.

| Component | Example | Meaning |
|-----------|---------|---------|
| `ActorID` | `1001` | Unique actor identifier |
| `SentenceCode` | `IEO` | The spoken sentence |
| `EmotionCode` | `HAP` | Emotion label — matches the subdirectory name |
| `LevelCode` | `HI` | Emotional intensity: `LO` (low), `MD` (medium), `HI` (high), `XX` (not applicable) |

`XX` is used exclusively for `NEU` (neutral), since neutral has no meaningful intensity variation — this is why NEU has fewer samples than the other classes. The subdirectory name is the class label used for training; the level code is available in the filename if you want to use intensity as an additional signal.

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

---

## MFCC classical ML baseline

Use these scripts when you want a non-deep-learning baseline using audio features such as MFCCs, deltas, chroma, spectral centroid, bandwidth, rolloff, zero-crossing rate, RMS, and duration.

All ML baseline code and generated ML artifacts live under:

```bash
SystemCode/data/training/ml/
```


### 1. Create and activate an environment

```bash
cd SystemCode/backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run a fast smoke test

This confirms the scripts and dataset path are working without training on the full dataset.

```bash
cd ../data/training/ml
python train_mfcc_model.py --limit-per-class 20 --model-name model_mfcc_smoke.joblib --force
```

### 3. Train the full MFCC SVM baseline

```bash
python train_mfcc_model.py --classifier svm --model-name model_mfcc_svm.joblib --cv
```

The script:

- Loads WAV files from `../../../datasets/train` and `../../../datasets/test`.
- Trims long silences and normalizes volume.
- Extracts MFCC-style summary features from each audio clip.
- Builds a speaker-safe validation split from the training set using CREMA-D actor IDs.
- Optionally runs grouped cross-validation with `--cv`.
- Saves the trained model and reports in `SystemCode/data/training/ml/`.

Generated artifacts:

```bash
ml/model_mfcc_svm.joblib
ml/mfcc_metrics.json
ml/mfcc_classification_report.txt
ml/mfcc_confusion_matrix.csv
ml/mfcc_train_*.npz
ml/mfcc_test_*.npz
```

### 4. Try one prediction

```bash
python predict_mfcc_model.py ../../../datasets/test/HAP/1001_IEO_HAP_HI.wav --model-path model_mfcc_svm.joblib
```

If that exact filename is not present, choose any `.wav` file from one of the `../../../datasets/test/<LABEL>/` folders.

### Useful options

```bash
# Random Forest alternative
python train_mfcc_model.py --classifier random_forest --model-name model_mfcc_rf.joblib

# Simulate telephone bandwidth during feature extraction
python train_mfcc_model.py --telephone-bandpass --model-name model_mfcc_phone_svm.joblib

# Rebuild feature caches after changing feature settings
python train_mfcc_model.py --force
```
