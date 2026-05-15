# Backend — Audio Emotion Analyzer API

FastAPI service that converts an uploaded audio clip into a mel-spectrogram and runs it through trained emotion classifiers.

## Run

From the repo root run `./start.sh` (on Windows, inside WSL). It creates `backend/.venv`, installs `requirements.txt`, and runs the server — there is no separate manual setup.

Server: http://localhost:8000 — interactive docs at http://localhost:8000/docs.

## Endpoints

- `GET /` — health check, returns `{"message": "Audio Emotion Analyzer API"}`.
- `GET /api/models` — list available models.
- `POST /api/analyze` — multipart: `audio` file + `models` form field, a JSON-encoded array of model ids (e.g. `["model_resnet50.pth"]`). Returns spectrogram data plus per-model emotion predictions.
- `POST /api/analyze-emotion` — multipart: `audio` file + optional `model_id`. Returns predictions for a single model.

Emotion labels: `ANG`, `DIS`, `FEA`, `HAP`, `NEU`, `SAD`.

## Structure

```
backend/app/
  main.py                       FastAPI app + CORS; entry point (python -m app.main)
  api/routes.py                 endpoints, model config, response shaping
  services/audio_processor.py   audio load + resample to 16 kHz
  services/spectrogram.py       mel-spectrogram + model input image
  models/predictors/
    registry.py                 model id -> predictor factory + cache
    base.py                     predictor interface
    resnet.py                   ResNet50 predictor
    yolo.py                     Ultralytics YOLO predictor
    device.py                   CUDA/CPU selection
```

Weights load from `SystemCode/data/training/` (`model_resnet50.pth`, `model_yolo.pt`) and are registered in `registry.py`; `model_resnet50.pth` is the default. Inference uses CUDA automatically when available, otherwise CPU.

## Pipeline

1. Load audio with librosa (any format), resample to 16 kHz, and trim leading/trailing silence.
2. Render a grayscale mel-spectrogram (512 mel bins) to a 640×640 RGB image — the model input.
3. Run the selected predictor: the ResNet path resizes to 224×224 and normalizes before inference; the YOLO path applies its own preprocessing. Softmax over the six emotion classes.
4. Return probabilities, the top emotion, and spectrogram data.
