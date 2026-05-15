# ResNet + YOLO Multi-Model Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the YOLO model a selectable, working option alongside ResNet, both fed by one spectrogram-image pipeline that exactly reproduces `data/prep/crema_d_spectrograms.ipynb`.

**Architecture:** A new shared `generate_model_input_image` renders audio to the exact prep PNG (16 kHz, silence-trimmed, `n_mels=512`, `hop=64`, matplotlib grayscale) as an in-memory PIL image. A predictor registry maps `model_id` to a `ResNetPredictor` or `YoloPredictor`, each applying its own thin preprocessing to that image. `/analyze-emotion` accepts a `model_id` form field and dispatches through the registry. Response JSON shape is unchanged.

**Tech Stack:** FastAPI, PyTorch/torchvision (ResNet), ultralytics (YOLO), librosa, matplotlib (Agg, object API), Pillow.

**Conventions (project):** No formal unit tests — manual verification only. No comments in generated code (the user adds their own). Follow existing repo module patterns.

---

### Task 1: Add backend dependencies

**Files:**
- Modify: `backend/requirements.txt`

- [ ] **Step 1: Append the new dependencies**

Add these lines to the end of `backend/requirements.txt`:

```
matplotlib==3.8.4
pillow==10.3.0
ultralytics==8.3.0
```

- [ ] **Step 2: Install into a clean environment and verify**

Run:

```bash
cd backend && python -m venv .venv-verify && . .venv-verify/bin/activate && pip install -r requirements.txt
```

Expected: install completes with no dependency-resolution error. `ultralytics` must coexist with the pinned `torch==2.2.2` (it does not force a torch upgrade). If pip reports an incompatibility, bump only `ultralytics` to the newest 8.3.x that accepts `torch==2.2.2` and re-run.

- [ ] **Step 3: Confirm imports load**

Run:

```bash
. backend/.venv-verify/bin/activate && python -c "import matplotlib, PIL, ultralytics; from ultralytics import YOLO; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 4: Remove the throwaway verify venv and commit**

```bash
rm -rf backend/.venv-verify
git add backend/requirements.txt
git commit -m "Chore. Add matplotlib, pillow, ultralytics for YOLO inference path."
```

---

### Task 2: Add the shared prep-faithful spectrogram-image function

**Files:**
- Modify: `backend/app/services/spectrogram.py`

- [ ] **Step 1: Add imports and the new function**

Add these imports at the top of `backend/app/services/spectrogram.py` (keep the existing `import librosa` / `import numpy as np`):

```python
import io
from PIL import Image
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
```

Append this function to the end of the file:

```python
MODEL_SR = 16000
MODEL_N_FFT = 2048
MODEL_HOP_LENGTH = 64
MODEL_N_MELS = 512
MODEL_TOP_DB = 20
MODEL_IMG_PX = 640
MODEL_DPI = 300


def generate_model_input_image(audio_data, sr):
    y = np.asarray(audio_data, dtype=np.float32)

    if sr != MODEL_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=MODEL_SR)
        sr = MODEL_SR

    trimmed, _ = librosa.effects.trim(y, top_db=MODEL_TOP_DB)
    if trimmed.size > 0:
        y = trimmed

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=MODEL_N_FFT,
        hop_length=MODEL_HOP_LENGTH,
        n_mels=MODEL_N_MELS,
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    fig = Figure(figsize=(MODEL_IMG_PX / MODEL_DPI, MODEL_IMG_PX / MODEL_DPI), dpi=MODEL_DPI)
    FigureCanvasAgg(fig)
    ax = fig.subplots(1, 1)
    ax.imshow(S_db, aspect="auto", origin="lower", cmap="gray")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=MODEL_DPI, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    return Image.open(buf).convert("RGB")
```

- [ ] **Step 2: Remove the obsolete matrix path**

Delete the entire `generate_spectrogram_for_model` function from `backend/app/services/spectrogram.py` (the old ResNet input path with `n_mels=128`, `max_time_steps`). Leave `generate_spectrogram` (the visualization heatmap) untouched.

- [ ] **Step 3: Manual verification against the prep notebook**

Run (uses any WAV present under `data/training/data/test/`; pick one path that exists):

```bash
cd backend && python -c "
import librosa, numpy as np
from app.services.spectrogram import generate_model_input_image
y, sr = librosa.load('../data/training/data/test/ANG/' + __import__('os').listdir('../data/training/data/test/ANG')[0], sr=16000)
img = generate_model_input_image(y, sr)
print('mode', img.mode, 'size', img.size)
img.save('/tmp/model_input_check.png')
"
```

Expected: prints `mode RGB size (W, H)` with W and H both near 640 (bbox_inches="tight" may shift by a few px — acceptable). Open `/tmp/model_input_check.png` and visually confirm it looks like a grayscale mel spectrogram (low frequencies at bottom, time on x-axis), matching the look of a PNG under `data/training/data/test/ANG/`.

- [ ] **Step 4: Commit**

```bash
git add backend/app/services/spectrogram.py
git commit -m "Feat. Shared prep-faithful model-input spectrogram image; drop old matrix path."
```

---

### Task 3: Create the predictor package (base + registry)

**Files:**
- Create: `backend/app/models/predictors/__init__.py`
- Create: `backend/app/models/predictors/base.py`
- Create: `backend/app/models/predictors/registry.py`

- [ ] **Step 1: Create the package init**

Create `backend/app/models/predictors/__init__.py` as an empty file.

- [ ] **Step 2: Create the predictor protocol**

Create `backend/app/models/predictors/base.py`:

```python
from typing import Protocol
from PIL import Image


class Predictor(Protocol):
    def predict(self, image: Image.Image) -> dict[str, float]:
        ...
```

- [ ] **Step 3: Create the registry**

Create `backend/app/models/predictors/registry.py`:

```python
from pathlib import Path
from app.models.predictors.resnet import ResNetPredictor
from app.models.predictors.yolo import YoloPredictor

_MODELS_DIR = Path(__file__).resolve().parents[4] / "data" / "training"

DEFAULT_MODEL_ID = "model_resnet50.pth"

_FACTORIES = {
    "model_resnet50.pth": lambda: ResNetPredictor(str(_MODELS_DIR / "model_resnet50.pth")),
    "model_yolo.pt": lambda: YoloPredictor(str(_MODELS_DIR / "model_yolo.pt")),
}

_CACHE = {}


def available_model_ids():
    return list(_FACTORIES.keys())


def get_predictor(model_id):
    if model_id not in _FACTORIES:
        raise KeyError(model_id)
    if model_id not in _CACHE:
        _CACHE[model_id] = _FACTORIES[model_id]()
    return _CACHE[model_id]
```

Note: `parents[4]` resolves `backend/app/models/predictors/registry.py` up to the `SystemCode` directory, so `_MODELS_DIR` is `SystemCode/data/training` — the same directory the existing `/models` endpoint scans.

- [ ] **Step 4: Commit**

```bash
git add backend/app/models/predictors/__init__.py backend/app/models/predictors/base.py backend/app/models/predictors/registry.py
git commit -m "Feat. Predictor protocol and lazy model registry."
```

(`registry.py` imports `resnet`/`yolo`, created next — do not import the registry until Task 5.)

---

### Task 4: Implement the ResNet and YOLO predictors

**Files:**
- Create: `backend/app/models/predictors/resnet.py`
- Create: `backend/app/models/predictors/yolo.py`

- [ ] **Step 1: Create the ResNet predictor**

Create `backend/app/models/predictors/resnet.py`:

```python
import os
import torch
import torch.nn as nn
from torchvision import models, transforms

EMOTION_LABELS = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
IMAGE_SIZE = 224

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class _EmotionClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = models.resnet50(weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


class ResNetPredictor:
    def __init__(self, weight_path):
        self._weight_path = weight_path
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        if not os.path.exists(self._weight_path):
            raise FileNotFoundError(self._weight_path)
        checkpoint = torch.load(self._weight_path, map_location=_DEVICE)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        model = _EmotionClassifier(len(EMOTION_LABELS))
        model.resnet.load_state_dict(state_dict)
        model.to(_DEVICE)
        model.eval()
        self._model = model

    def predict(self, image):
        self._load()
        tensor = _TRANSFORM(image).unsqueeze(0).to(_DEVICE)
        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return {label: float(p) for label, p in zip(EMOTION_LABELS, probs)}
```

- [ ] **Step 2: Create the YOLO predictor**

Create `backend/app/models/predictors/yolo.py`:

```python
import os
from ultralytics import YOLO

EMOTION_LABELS = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]


class YoloPredictor:
    def __init__(self, weight_path):
        self._weight_path = weight_path
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        if not os.path.exists(self._weight_path):
            raise FileNotFoundError(self._weight_path)
        self._model = YOLO(self._weight_path)

    def predict(self, image):
        self._load()
        result = self._model.predict(image, verbose=False)[0]
        names = result.names
        probs = result.probs.data.tolist()
        scored = {names[i]: float(p) for i, p in enumerate(probs)}
        return {label: scored.get(label, 0.0) for label in EMOTION_LABELS}
```

- [ ] **Step 3: Manual verification of both predictors**

Run (reuses a real spectrogram image via the Task 2 function):

```bash
cd backend && python -c "
import os, librosa
from app.services.spectrogram import generate_model_input_image
from app.models.predictors.registry import get_predictor, available_model_ids
wav_dir = '../data/training/data/test/HAP'
y, sr = librosa.load(os.path.join(wav_dir, os.listdir(wav_dir)[0]), sr=16000)
img = generate_model_input_image(y, sr)
for mid in available_model_ids():
    out = get_predictor(mid).predict(img)
    print(mid, 'sum=%.3f' % sum(out.values()), out)
"
```

Expected: each model prints a 6-key dict over `ANG/DIS/FEA/HAP/NEU/SAD` whose values sum to ~1.000. Predictions need not agree, but should be a valid distribution (no NaN, no all-zero).

- [ ] **Step 4: Commit**

```bash
git add backend/app/models/predictors/resnet.py backend/app/models/predictors/yolo.py
git commit -m "Feat. ResNet and YOLO predictors with training-matched preprocessing."
```

---

### Task 5: Wire the registry into routes and remove the old model module

**Files:**
- Modify: `backend/app/api/routes.py`
- Delete: `backend/app/models/emotion_model.py`

- [ ] **Step 1: Replace imports and model config in `routes.py`**

In `backend/app/api/routes.py`, replace lines 7-9:

```python
from app.services.audio_processor import process_audio
from app.services.spectrogram import generate_spectrogram, generate_spectrogram_for_model
from app.models.emotion_model import predict_emotion
```

with:

```python
from app.services.audio_processor import process_audio
from app.services.spectrogram import generate_spectrogram, generate_model_input_image
from app.models.predictors.registry import get_predictor, available_model_ids, DEFAULT_MODEL_ID
```

Replace the `MODEL_CONFIG` block (lines 14-20) with:

```python
MODEL_CONFIG = {
    "model_resnet50.pth": {
        "id": "model_resnet50.pth",
        "name": "ResNet50 v1",
        "description": "Pretrained ResNet50 on CREMA-D dataset"
    },
    "model_yolo.pt": {
        "id": "model_yolo.pt",
        "name": "YOLO11s-cls v1",
        "description": "Ultralytics YOLO11s classifier on CREMA-D dataset"
    }
}
```

- [ ] **Step 2: Make `/analyze-emotion` model-aware**

Replace the entire `analyze_emotion` function (the `@router.post("/analyze-emotion")` handler) with:

```python
@router.post("/analyze-emotion")
async def analyze_emotion(
    audio: UploadFile = File(...),
    model_id: str = Form(None),
):
    """
    Analyze audio file and return emotion predictions and spectrogram data
    """
    selected_id = model_id or DEFAULT_MODEL_ID
    if selected_id not in available_model_ids():
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model_id '{selected_id}'. Available: {available_model_ids()}",
        )

    audio_bytes = await audio.read()
    audio_data, sr = process_audio(audio_bytes, filename=audio.filename)
    duration = len(audio_data) / sr

    spectrogram_data = generate_spectrogram(audio_data, sr)
    model_image = generate_model_input_image(audio_data, sr)

    try:
        emotion_probs = get_predictor(selected_id).predict(model_image)
    except FileNotFoundError as exc:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail=f"Model weights not available for '{selected_id}': {exc}",
        )

    top_emotion = max(emotion_probs, key=emotion_probs.get)
    top_emotion_name = localize_emotion_label(top_emotion)
    confidence = float(emotion_probs[top_emotion])

    return {
        "duration": float(duration),
        "sampleRate": sr,
        "spectrogramData": spectrogram_data.tolist(),
        "emotions": emotion_probs,
        "emotionDistribution": [
            {"emotion": localize_emotion_label(label), "label": label, "confidence": float(prob)}
            for label, prob in emotion_probs.items()
        ],
        "topEmotion": top_emotion,
        "topEmotionName": top_emotion_name,
        "confidence": confidence,
        "segments": [
            {
                "start": 0.0,
                "end": round(duration, 2),
                "emotion": top_emotion_name,
                "confidence": confidence,
            }
        ],
    }
```

- [ ] **Step 3: Make `/analyze` dispatch per model**

In the `analyze_audio` function, replace the `else:` branch inside the `for model_id in model_ids:` loop (the block currently calling `generate_spectrogram_for_model` / `predict_emotion`) with:

```python
        else:
            spectrogram_data = generate_spectrogram(audio_data, sr)
            model_image = generate_model_input_image(audio_data, sr)
            resolved_id = model_id if model_id in available_model_ids() else DEFAULT_MODEL_ID
            emotion_probs = get_predictor(resolved_id).predict(model_image)
            top_emotion = max(emotion_probs, key=emotion_probs.get)
            confidence = float(emotion_probs[top_emotion])
            segments = [
                {
                    "start": 0.0,
                    "end": round(duration, 2),
                    "emotion": localize_emotion_label(top_emotion),
                    "confidence": round(confidence, 2),
                }
            ]
```

- [ ] **Step 4: Delete the obsolete model module and dead helper**

Delete `backend/app/models/emotion_model.py`. In `routes.py`, delete the now-unused `predict_emotions` function (the one at the bottom that calls `generate_spectrogram_for_model`).

- [ ] **Step 5: Manual verification — server boots and both models respond**

Start the server:

```bash
cd backend && uvicorn app.main:app --port 8000 &
sleep 8
```

Pick a real test WAV and call the endpoint for each model:

```bash
W=$(ls ../data/training/data/test/SAD/*.wav | head -1)
curl -s -F "audio=@$W" -F "model_id=model_resnet50.pth" http://localhost:8000/api/analyze-emotion | python -c "import sys,json; d=json.load(sys.stdin); print('resnet', d['topEmotion'], round(d['confidence'],3))"
curl -s -F "audio=@$W" -F "model_id=model_yolo.pt" http://localhost:8000/api/analyze-emotion | python -c "import sys,json; d=json.load(sys.stdin); print('yolo', d['topEmotion'], round(d['confidence'],3))"
curl -s -F "audio=@$W" -F "model_id=bogus" http://localhost:8000/api/analyze-emotion | python -c "import sys,json; print('bad-id status:', json.load(sys.stdin))"
curl -s http://localhost:8000/api/models | python -c "import sys,json; print([m['id'] for m in json.load(sys.stdin)['models']])"
kill %1
```

Expected: `resnet` and `yolo` lines each print a label from `ANG/DIS/FEA/HAP/NEU/SAD` with a confidence; the `bogus` call returns a 400 detail listing available models; `/api/models` lists both `model_resnet50.pth` and `model_yolo.pt`.

- [ ] **Step 6: Commit**

```bash
git add backend/app/api/routes.py
git rm backend/app/models/emotion_model.py
git commit -m "Feat. Route inference through model registry; model_id on /analyze-emotion."
```

---

### Task 6: Wire model selection through the frontend

**Files:**
- Modify: `frontend/src/utils/api.js`
- Modify: `frontend/src/components/AudioEmotionAnalyzer.js`

- [ ] **Step 1: Send `model_id` from `api.js`**

In `frontend/src/utils/api.js`, replace the `analyzeEmotion` function with:

```javascript
export async function analyzeEmotion(audioFile, modelId) {
  const formData = new FormData();
  formData.append("audio", audioFile);
  if (modelId) {
    formData.append("model_id", modelId);
  }

  const response = await fetch(`${API_BASE_URL}/analyze-emotion`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error("Backend request failed");
  }

  const data = await response.json();
  return data;
}
```

In the same file, replace the two hardcoded fallback arrays (inside `fetchAvailableModels`, in the `if (!data.models ...)` branch and the `catch` branch) with this list in both places:

```javascript
    return [
      {
        id: "model_resnet50.pth",
        name: "ResNet50 v1",
        description: "Pretrained ResNet50 on CREMA-D dataset",
      },
      {
        id: "model_yolo.pt",
        name: "YOLO11s-cls v1",
        description: "Ultralytics YOLO11s classifier on CREMA-D dataset",
      },
    ];
```

- [ ] **Step 2: Pass the selected model from the component**

In `frontend/src/components/AudioEmotionAnalyzer.js`, in `fetchEmotionAnalysisFromBackend`, change the call:

```javascript
    const results = await analyzeEmotion(audioFile);
```

to:

```javascript
    const results = await analyzeEmotion(audioFile, selectedModel);
```

- [ ] **Step 3: Manual end-to-end verification**

Start backend (`cd backend && uvicorn app.main:app --port 8000`) and frontend (`cd frontend && npm run dev`). In the browser: upload an audio clip, pick **ResNet50 v1** in the dropdown, click Analyze, note the top emotion; then pick **YOLO11s-cls v1**, Analyze the same clip again.

Expected: both produce results; the dropdown now lists both models; switching the model and re-analyzing visibly re-runs inference (results/JSON differ between the two models for at least some clips). Confirm in the browser network tab that the `/analyze-emotion` request includes the `model_id` form field matching the dropdown.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/utils/api.js frontend/src/components/AudioEmotionAnalyzer.js
git commit -m "Feat. Send selected model_id to backend; list YOLO in model fallback."
```

---

### Task 7: Final integration pass

**Files:** none (verification only)

- [ ] **Step 1: Clean-environment smoke test**

In a fresh shell:

```bash
cd backend && python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && uvicorn app.main:app --port 8000 &
sleep 10
W=$(ls ../data/training/data/test/NEU/*.wav | head -1)
for M in model_resnet50.pth model_yolo.pt; do
  curl -s -F "audio=@$W" -F "model_id=$M" http://localhost:8000/api/analyze-emotion \
    | python -c "import sys,json; d=json.load(sys.stdin); s=sum(d['emotions'].values()); print('$M', d['topEmotion'], 'sum=%.3f'%s)"
done
kill %1
```

Expected: both models return a valid distribution (`sum≈1.000`) and a top emotion. This confirms `requirements.txt` alone is sufficient to run the full path.

- [ ] **Step 2: Verify the dead matrix path is fully gone**

Run:

```bash
grep -rn "generate_spectrogram_for_model\|emotion_model\|predict_emotion" backend/app
```

Expected: no matches (every reference to the old ResNet matrix path and old module is removed).

- [ ] **Step 3: Final commit if anything was adjusted**

```bash
git add -A && git commit -m "Chore. Final integration cleanup for multi-model inference." || echo "nothing to commit"
```

---

## Notes for the implementer

- `process_audio` already returns audio at 16 kHz (its `target_sr` default), so the resample guard in `generate_model_input_image` is normally a no-op but stays for correctness if a caller passes another rate.
- The matplotlib **object API** (`Figure` + `FigureCanvasAgg`, no `pyplot`) is deliberate: FastAPI runs sync handlers in a threadpool and `pyplot` global state is not thread-safe.
- Class order is alphabetical for both models (`ImageFolder.class_to_idx` and ultralytics `result.names`), matching `["ANG","DIS","FEA","HAP","NEU","SAD"]`; the YOLO predictor still remaps via `result.names` defensively.
- Manual sanity goal: running each model on a few labelled CREMA-D test clips should give plausible predictions, confirming the corrected ResNet input path and the YOLO path both match training.
- Verification commands assume the CREMA-D dataset is present under `data/training/data/{train,test}/<EMOTION>/`. If it is not (it is gitignored / downloaded separately), substitute any short speech WAV for the `audio=@...` / `librosa.load` paths — schema and distribution checks still hold; only the "predictions are plausible" check needs labelled clips.
