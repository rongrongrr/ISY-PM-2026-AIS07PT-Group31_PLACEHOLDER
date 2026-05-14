# Backend - Audio Emotion Analyzer API

FastAPI-based backend for audio emotion classification using spectrograms and deep learning models.

## Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the Server](#running-the-server)
- [API Endpoints](#api-endpoints)
- [Model Integration](#model-integration)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# 1. Navigate to backend
cd SystemCode/backend

# 2. Create & activate virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# OR
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
python -m app.main
```

✅ Server running at: **http://localhost:8000**

---

## Project Structure

```
backend/
├── app/
│   ├── main.py                    ← Entry point (Uvicorn server)
│   ├── api/
│   │   └── routes.py              ← API endpoints & request handling
│   ├── services/
│   │   ├── audio_processor.py     ← Audio loading & resampling
│   │   └── spectrogram.py         ← Mel-spectrogram generation
│   └── models/
│       └── emotion_model.py       ← Model inference (dummy or real)
├── requirements.txt               ← Python dependencies
└── README.md                      ← This file
```

---

## Setup & Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### 1. Create Virtual Environment

```bash
python -m venv venv
```

### 2. Activate Virtual Environment

**macOS/Linux:**

```bash
source venv/bin/activate
```

**Windows:**

```bash
venv\Scripts\activate
```

You should see `(venv)` prefix in your terminal.

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Installed packages:**

- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `python-multipart` - File upload support
- `torch` - Deep learning framework
- `librosa` - Audio processing
- `numpy` - Numerical operations
- `scipy` - Signal processing

### 4. Verify Installation

```bash
python -c "import fastapi; import librosa; import torch; print('✅ All dependencies installed!')"
```

---

## Running the Server

### Start Server

```bash
python -m app.main
```

**Expected Output:**

```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

### Access Server

- **API Root**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs (interactive Swagger UI)
- **Alternative Docs**: http://localhost:8000/redoc

### Stop Server

Press `Ctrl+C` in the terminal.

---

## API Endpoints

### Health Check

**GET** `/`

Returns API status.

```bash
curl http://localhost:8000
```

**Response:**

```json
{
  "message": "Audio Emotion Analyzer API"
}
```

---

### Analyze Emotion

**POST** `/api/analyze-emotion`

Analyzes audio file and predicts emotions.

**Request:**

```bash
curl -X POST http://localhost:8000/api/analyze-emotion \
  -F "audio=@path/to/audio.wav"
```

**Request Body:**

- `audio` (file, required) - Audio file (mp3, wav, m4a, ogg, flac, etc.)

**Response:**

```json
{
  "emotions": {
    "ANG": 0.05,
    "DIS": 0.08,
    "FEA": 0.12,
    "HAP": 0.55,
    "NEU": 0.15,
    "SAD": 0.05
  },
  "top_emotion": "HAP",
  "confidence": 0.55,
  "spectrogram": "base64_encoded_image_data"
}
```

---

## Model Integration

### Current Status: Dummy Mode

The backend is configured to return **fake emotion predictions** for testing and development.

This allows you to:

- ✅ Test the full frontend-backend integration
- ✅ Verify API endpoints work correctly
- ✅ Develop UI without a trained model

### Enable Real Model Mode

When you have a trained model (`data/training/model.pth`), enable real inference:

**File:** `app/api/routes.py`

Find line 15:

```python
USE_DUMMY_MODELS = True  # Set to False when real models are ready
```

Change to:

```python
USE_DUMMY_MODELS = False  # Now using real models!
```

Then restart the server:

```bash
# Press Ctrl+C to stop
python -m app.main
```

---

### Load Your Trained Model

Update the `predict_emotion()` function in `app/models/emotion_model.py`:

```python
import torch
import torchvision.models as models
import torch.nn as nn
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent.parent / "data" / "training" / "model.pth"

def predict_emotion(spectrogram_input):
    """
    Load trained ResNet50 model and predict emotions from spectrogram.

    Args:
        spectrogram_input: Mel-spectrogram array (normalized, 3-channel, 224×224)

    Returns:
        dict: Emotion probabilities {emotion: probability, ...}
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    # Reconstruct model (must match training architecture)
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    # Prepare input tensor (add batch dimension: 1 × 3 × 224 × 224)
    spectrogram_tensor = torch.from_numpy(spectrogram_input).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(spectrogram_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    # Map to emotion labels
    CLASS_NAMES = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
    return {emotion: float(prob) for emotion, prob in zip(CLASS_NAMES, probabilities)}
```

---

### Model Checkpoint Structure

Your saved model should contain:

```python
{
    "model_state_dict": <pytorch state dict>,
    "class_names": ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"],
    "image_size": 224,
    "num_classes": 6
}
```

This matches the checkpoint saved by [training_resnet50.ipynb](../data/training/training_resnet50.ipynb).

---

## Audio Processing Pipeline

When you send audio to `/api/analyze-emotion`:

```
1. Load Audio
   └─ Librosa loads audio file (any format)
   └─ Resample to 16 kHz (standard for speech)

2. Generate Spectrogram
   └─ Compute Mel-frequency spectrogram
   └─ Normalize to 0-1 range
   └─ Convert grayscale to 3-channel (for ResNet50)
   └─ Resize to 224×224

3. Model Inference
   └─ Forward pass through ResNet50
   └─ Apply softmax to get probabilities
   └─ Return emotion predictions

4. Format Response
   └─ Emotion probabilities
   └─ Top emotion with confidence
   └─ Encoded spectrogram image
```

---

## Directory Structure Details

### `app/main.py`

Entry point. Creates FastAPI app and starts Uvicorn server.

```python
python -m app.main    # Run from backend/ directory
```

### `app/api/routes.py`

Defines all API endpoints:

- `GET /` - Health check
- `POST /api/analyze-emotion` - Emotion classification

Also contains:

- `USE_DUMMY_MODELS` flag (line 15)
- Request validation
- Response formatting

### `app/services/audio_processor.py`

Handles audio loading and preprocessing:

- Loads audio files (mp3, wav, m4a, etc.)
- Resamples to 16 kHz
- Validates audio duration

### `app/services/spectrogram.py`

Generates mel-spectrograms:

- Computes mel-frequency spectrogram
- Normalizes to 0-1 range
- Converts to 3-channel RGB (for CNN compatibility)
- Resizes to 224×224 pixels

### `app/models/emotion_model.py`

Model inference logic:

- `predict_emotion()` - Runs model on spectrogram
- Currently returns dummy data
- Implement with real model when ready

---

## Troubleshooting

### Port 8000 Already in Use

**Error:** `Address already in use`

**Solution 1: Kill Process**

macOS/Linux:

```bash
lsof -ti:8000 | xargs kill -9
```

Windows:

```bash
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**Solution 2: Use Different Port**

Edit `app/main.py`:

```python
uvicorn.run(app, host="0.0.0.0", port=8001)  # Changed from 8000
```

---

### No Module Named 'app'

**Error:** `ModuleNotFoundError: No module named 'app'`

**Solution:**

```bash
# Make sure you're in the backend directory
cd SystemCode/backend

# Run with python -m (not just python main.py)
python -m app.main
```

---

### Module Not Found Errors

**Error:** `ModuleNotFoundError: No module named 'librosa'` (or other packages)

**Solution:**

```bash
# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt

# Verify installation
python -c "import librosa; print('✅ Librosa installed')"
```

---

### CORS Errors (Frontend Can't Connect)

**Error:** `Access to XMLHttpRequest blocked by CORS policy`

**Solution:** Verify `app/main.py` has CORS configuration:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

If frontend is on different port, update `allow_origins`.

---

### GPU/CUDA Issues

**Error:** `RuntimeError: CUDA out of memory` or CUDA-related errors

**Solution:** Model automatically falls back to CPU if CUDA unavailable:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

No action needed - CPU inference works fine for development.

---

### Audio File Not Processing

**Error:** `Failed to decode audio` or `No audio frames found`

**Possible causes:**

1. Corrupted audio file
2. Unsupported format
3. Missing audio data

**Solution:**

```bash
# Test with a known-good audio file (WAV, MP3)
# Check file with system commands
file path/to/audio.wav
```

---

### Model File Not Found

**Error:** `FileNotFoundError: [Errno 2] No such file or directory: 'model.pth'`

**Solution:**

1. Train model first: See [training/README.md](../training/README.md)
2. Verify file exists: `SystemCode/data/training/model.pth`
3. Keep `USE_DUMMY_MODELS = True` until model is ready

---

## Quick Reference

| Task           | Command                                                                       |
| -------------- | ----------------------------------------------------------------------------- |
| Activate env   | `source venv/bin/activate` (macOS/Linux) or `venv\Scripts\activate` (Windows) |
| Install deps   | `pip install -r requirements.txt`                                             |
| Start server   | `python -m app.main`                                                          |
| Stop server    | `Ctrl+C`                                                                      |
| View API docs  | Open http://localhost:8000/docs                                               |
| Test endpoint  | `curl http://localhost:8000`                                                  |
| Deactivate env | `deactivate`                                                                  |

---

## Next Steps

1. ✅ Train emotion model: See [training/README.md](../training/README.md)
2. ✅ Save checkpoint to `data/training/model.pth`
3. ✅ Update `predict_emotion()` function
4. ✅ Set `USE_DUMMY_MODELS = False`
5. ✅ Restart server and test with real audio

---

## Support

- **API Docs**: http://localhost:8000/docs
- **FastAPI Docs**: https://fastapi.tiangolo.com
- **Librosa Docs**: https://librosa.org
- **PyTorch Docs**: https://pytorch.org/docs
