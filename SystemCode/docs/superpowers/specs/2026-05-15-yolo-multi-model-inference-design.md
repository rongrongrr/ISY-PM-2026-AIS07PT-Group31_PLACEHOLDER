# Multi-model inference (ResNet + YOLO) on a shared prep-faithful input

Date: 2026-05-15
Status: Approved

## Problem

The frontend offers a model dropdown, but the selected model never reaches the
backend. The live "Analyze Emotions" button calls `/analyze-emotion`, which
takes no model parameter and hardcodes the ResNet path. A second YOLO
classification model (`data/training/model_yolo.pt`, an ultralytics YOLO11s-cls
checkpoint) now exists and must be a selectable, working option.

A deeper issue surfaced during exploration: both models were trained on the
**same** spectrogram PNGs produced by `data/prep/crema_d_spectrograms.ipynb`
(ResNet via `torchvision.ImageFolder`, YOLO via ultralytics). The current
backend ResNet path recomputes a mel matrix with different parameters
(`n_mels=128`, `hop=512`, `[0,1]`-normalized, interpolated) that does **not**
match training — ResNet is running on out-of-distribution input today. The fix
must unify both models on one prep-faithful input pipeline.

## Goals

- YOLO selectable in the dropdown and producing valid predictions.
- Selected model ID actually drives backend inference.
- One shared spectrogram-image function reproducing the prep notebook exactly,
  feeding both models.
- Correct the existing out-of-distribution ResNet path as part of this work.
- No change to the response JSON shape (frontend result components untouched).

## Non-goals (YAGNI)

Multi-model side-by-side comparison, per-segment temporal emotion, model-upload
UI, GPU batching, caching of rendered spectrograms.

## Source of truth: the prep pipeline

`data/prep/crema_d_spectrograms.ipynb` `wav_to_spectrogram_png`:

- `librosa.load(wav, sr=16000)`
- `librosa.effects.trim(y, top_db=20)`
- `librosa.feature.melspectrogram(y, sr, n_fft=2048, hop_length=64, n_mels=512)`
- `librosa.power_to_db(S, ref=np.max)`
- matplotlib render: `imshow(S_db, aspect="auto", origin="lower", cmap="gray")`,
  axes off, `figsize=(640/300, 640/300)`, `dpi=300`,
  `subplots_adjust(left=0,right=1,top=1,bottom=0)`,
  `savefig(bbox_inches="tight", pad_inches=0)`.

Per-model preprocessing applied to that PNG:

- ResNet (`training_resnet50.ipynb` cell 10): `Resize((224,224))` →
  `Grayscale(num_output_channels=3)` → `ToTensor()` →
  `Normalize(mean=[.5,.5,.5], std=[.5,.5,.5])`.
- YOLO: ultralytics handles its own 640 resize/normalization internally.

Class order is alphabetical for both (`ImageFolder.class_to_idx` and
ultralytics `r.names`): `["ANG","DIS","FEA","HAP","NEU","SAD"]`.

## Architecture

```
audio bytes ─► process_audio ─► (audio_data, sr)
                                   │
      ┌────────────────────────────┴───────────────────┐
      ▼                                                  ▼
generate_spectrogram (UNCHANGED)        generate_model_input_image  [NEW]
 → [0,1] matrix, n_mels=128              exact port of wav_to_spectrogram_png
 → frontend heatmap only                 → in-memory PNG → PIL.Image
                                                          │
                                      ┌───────────────────┴──────────────────┐
                                      ▼                                        ▼
                              ResNetPredictor                          YoloPredictor  [NEW]
                              224 + Gray(3) + .5/.5                     YOLO.predict
                              → softmax                                  → r.probs / r.names
                                      └───────────────────┬──────────────────┘
                                                          ▼
                                  {ANG,DIS,FEA,HAP,NEU,SAD: prob}  (shape unchanged)
```

A predictor registry maps `model_id → predictor`; the endpoint resolves the
selected model and dispatches. Response JSON is unchanged.

## Components and file changes

**`backend/app/services/spectrogram.py`**
- Add `generate_model_input_image(audio_data, sr) -> PIL.Image`: verbatim port
  of `wav_to_spectrogram_png`, using the object-oriented matplotlib API
  (`matplotlib.figure.Figure` + `FigureCanvasAgg`, **no `pyplot`**) for
  thread-safety under FastAPI's sync threadpool; render to `BytesIO` then
  `PIL.Image.open`. Resample to 16 kHz if `sr != 16000`.
- Keep `generate_spectrogram` (visualization heatmap) unchanged.
- Remove `generate_spectrogram_for_model` (the out-of-distribution matrix path).

**`backend/app/models/predictors/` (new package)**
- `base.py` — `Predictor` protocol: `predict(image: PIL.Image) -> dict[str,float]`.
- `resnet.py` — wraps existing `EmotionClassifier`; applies the correct
  training transforms (224 → Grayscale(3) → ToTensor → Normalize .5/.5);
  lazy-loads `model_resnet50.pth` (handles `model_state_dict` key).
- `yolo.py` — lazy `YOLO("model_yolo.pt")`; `.predict(image)`; reads
  `r.probs.data` mapped through `r.names`.
- `registry.py` — `{"model_resnet50.pth": ResNetPredictor,
  "model_yolo.pt": YoloPredictor}`, each lazy-loaded and cached on first use.

**`backend/app/models/emotion_model.py`**
- Removed; the resnet loader and (corrected) inference logic move into
  `predictors/resnet.py`.

**`backend/app/api/routes.py`**
- Add `model_yolo.pt` to `MODEL_CONFIG` so `/models` lists it (the `.pth` glob
  misses `.pt`; explicit config required).
- `/analyze-emotion` gains optional `model_id: str = Form(None)` (absent →
  ResNet default); builds the model input image once, calls
  `registry[model_id].predict(image)`.
- `/analyze` dispatches per model the same way (kept consistent though the UI
  does not use it).

**Frontend**
- `utils/api.js`: `analyzeEmotion(audioFile, modelId)` appends `model_id`;
  extend the hardcoded fallback model list to include YOLO.
- `components/AudioEmotionAnalyzer.js`: pass `selectedModel` to
  `analyzeEmotion`.
- `components/ModelSelection.js`: unchanged.

**`backend/requirements.txt`**
- Add `ultralytics`, `matplotlib`, `pillow` (matplotlib is currently not a
  backend dependency). Pin versions compatible with the existing
  `torch==2.2.2`; verify a clean-venv install during implementation.

## Data flow and error handling

- Absent `model_id` → ResNet default (backward compatible).
- Unknown `model_id` → HTTP 400 listing available IDs (no silent fallback, so
  frontend wiring bugs are not masked).
- Missing model weight file or predictor load failure → HTTP 503 with a clear
  message; not swallowed into dummy predictions.
- `USE_DUMMY_MODELS` flag retained for offline frontend development.
- Silence-trim edge case: if `librosa.effects.trim` yields near-empty audio,
  fall back to untrimmed audio for the render (mirrors prep robustness).
- `duration` and the visualization heatmap use the original (untrimmed) audio;
  trimming applies only to the model input.

## Verification (manual — no formal test suite)

This is school coursework, not production; no unit/integration test suite.
Manual sanity checks during implementation:

- Render `generate_model_input_image` for one sample CREMA-D WAV and visually
  compare against the PNG the prep notebook produces for the same file.
- Hit `/analyze-emotion` with `model_id=model_resnet50.pth` and
  `=model_yolo.pt`; confirm the JSON schema is unchanged and each returns a
  normalized 6-key distribution.
- Run both models on a few labelled CREMA-D test clips; predictions should be
  sensible (confirms the corrected ResNet path and the YOLO path match
  training).
- Confirm a clean-venv `pip install -r requirements.txt` succeeds.

## Conventions

Generated code contains **no comments** (the user adds explanatory comments
afterward). Follow existing module/structure patterns in the repo.
