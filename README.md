# Audio Emotion Analyzer - Setup Guide

A full-stack application for analyzing emotional patterns in speech using AI models. Upload audio files or record directly, then visualize emotions on spectrograms.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Backend Setup](#backend-setup)
- [Frontend Setup](#frontend-setup)
- [Running the Application](#running-the-application)
- [Testing the Application](#testing-the-application)
- [Switching from Dummy to Real Models](#switching-from-dummy-to-real-models)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

- **Python 3.8+** (for backend)
- **Node.js 16+** and **npm** (for frontend)
- **Git** (for version control)

### Check Versions

```bash
python --version  # Should be 3.8 or higher
node --version    # Recommended: v18 or higher (Node 16 may work but recent toolchain/plugins are tested with Node 18+)
npm --version     # Should be 8 or higher
```

---

## Project Structure

```
SystemCode/
├── backend/
│   ├── app/
│   │   ├── main.py
+│   │   ├── api/
│   │   │   └── routes.py
│   │   ├── services/
│   │   │   ├── audio_processor.py
│   │   │   └── spectrogram.py
│   │   └── models/
│   │       ├── baseline.py
│   │       ├── advanced.py
│   │       └── ensemble.py
│   ├── requirements.txt
│   └── README.md
│
└── frontend/
    ├── src/
    │   ├── components/
    │   │   └── AudioEmotionAnalyzer.js
    │   ├── utils/
    │   │   └── api.js
    │   ├── App.js
    │   ├── main.js
    │   └── index.css
    ├── package.json
    ├── vite.config.js
    ├── tailwind.config.js
    └── postcss.config.js
```

Note: The backend now uses PEP 420 namespace packages (there are intentionally no empty `__init__.py` files in the `app` tree). If you see "No module named 'app'" errors, run the backend from the `SystemCode/backend` directory using `python -m app.main`, or ensure the `backend` folder is on your `PYTHONPATH`.

---

## Backend Setup

### 1. Navigate to Backend Directory

```bash
cd SystemCode/backend
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt.

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:

- FastAPI (web framework)
- Uvicorn (ASGI server)
- Librosa (audio processing)
- NumPy (numerical operations)
- PyTorch (ML framework)
- And other required packages

### 4. Verify Installation

```bash
python -c "import fastapi; import librosa; import numpy; print('All packages installed successfully!')"
```

---

## Frontend Setup

### 1. Navigate to Frontend Directory

Open a **NEW terminal** window/tab and run:

```bash
cd SystemCode/frontend
```

### 2. Install Dependencies

```bash
npm install
```

This will install:

- React (UI framework)
- Vite (build tool)
- Tailwind CSS (styling)
- Lucide React (icons)
- And other required packages

### 3. Verify Installation

```bash
npm list react vite tailwindcss
```

You should see the installed versions listed.

---

## Running the Application

You need **TWO terminal windows** - one for backend, one for frontend.

### Terminal 1: Start Backend

```bash
cd SystemCode/backend

# Activate virtual environment (if not already activated)
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows

# Run the server
python -m app.main
```

**Expected Output:**

```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [XXXXX] using WatchFiles
INFO:     Started server process [XXXXX]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

✅ Backend is running at: **http://localhost:8000**

### Terminal 2: Start Frontend

```bash
cd SystemCode/frontend

# Run the dev server
npm run dev
```

**Expected Output:**

```
  VITE v7.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
  ➜  press h + enter to show help
```

✅ Frontend is running at: **http://localhost:5173**

---

## End-to-End Workflow

### Step 1: Prepare Backend (Terminal 1)

```bash
cd SystemCode/backend
source venv/bin/activate        # macOS/Linux or venv\Scripts\activate on Windows
python -m app.main
```

Wait for: `Application startup complete`

### Step 2: Prepare Frontend (Terminal 2)

```bash
cd SystemCode/frontend
npm run dev
```

Wait for: `Local: http://localhost:5173/`

### Step 3: Use the Application

1. Open browser → **http://localhost:5173**
2. Click **"Upload Audio File"** or **"Record Audio"**
3. Select audio or record your voice
4. Click **"Analyze Emotions"**
5. View results:
   - Top emotion with confidence %
   - Emotion probabilities (bar chart)
   - Spectrogram visualization

### Step 4: Enable Real Model (When Ready)

Once you've trained your model and saved it as `data/training/model.pth`:

**Edit:** `SystemCode/backend/app/api/routes.py` (line 15)

```python
USE_DUMMY_MODELS = False  # Change from True
```

Then restart backend:

```bash
# Stop: Ctrl+C
# Restart: python -m app.main
```

---

## Data Flow Diagram

```
┌─────────────────────────────┐
│      Frontend (React)       │
│   http://localhost:5173     │
└──────────────┬──────────────┘
               │
        Audio Blob (.mp3/.wav)
               │
               ▼
   ┌───────────────────────┐
   │  Backend (FastAPI)    │
   │ http://localhost:8000 │
   └──────────┬────────────┘
              │
              ├─ 1. Load audio bytes
              ├─ 2. Resample to 16 kHz
              ├─ 3. Generate mel-spectrogram
              └─ 4. Run inference
              │
              ▼
   ┌───────────────────────┐
   │   ResNet50 Model      │
   │    model.pth          │
   │  (Trained on CREMA-D) │
   └──────────┬────────────┘
              │
         Softmax Probabilities
              │
              ▼
    JSON Response
  {
    emotions: {
      ANG: 0.05,
      DIS: 0.08,
      FEA: 0.12,
      HAP: 0.55,
      NEU: 0.15,
      SAD: 0.05
    },
    top_emotion: "HAP",
    confidence: 0.55,
    spectrogram: "base64_image"
  }
```

---

## Testing the Application

### 1. Open Browser

Navigate to: **http://localhost:5173**

### 2. Verify Backend Connection

First, test the backend API directly:

- Open: **http://localhost:8000**
- You should see: `{"message": "Audio Emotion Analyzer API"}`

### 3. Test the UI

#### Upload Audio File

1. Click **"Upload Audio File"**
2. Select an audio file (`.mp3`, `.wav`, `.m4a`, etc.)
3. Select one or more models (checkboxes)
4. Click **"Generate Spectrograms"**

#### Record Audio

1. Click **"Record Audio"**
2. Allow microphone permissions when prompted
3. Speak for a few seconds
4. Click **"Stop Recording"**
5. Select models and click **"Generate Spectrograms"**

### 4. Expected Results

- Loading indicator appears
- Spectrogram visualization displays for each selected model
- Emotion labels appear on the x-axis timeline
- Different colored segments show different emotions

---

## Switching from Dummy to Real Models

Currently, the backend returns **dummy data** (random spectrograms and emotions). When you're ready to use real ML models:

### 1. Edit `backend/app/api/routes.py`

Find this line near the top:

```python
USE_DUMMY_MODELS = True  # Set to False when real models are ready
```

Change it to:

```python
USE_DUMMY_MODELS = False  # Now using real models!
```

### 2. Implement Real Models

Edit the `predict_emotions()` function in `backend/app/api/routes.py`:

```python
def predict_emotions(audio_data, sr, model_id):
    """
    Implement actual emotion prediction with real models
    """
    # Load your trained model
    model = load_model(model_id)

    # Extract features
    features = extract_features(audio_data, sr)

    # Make predictions
    predictions = model.predict(features)

    # Format results
    emotions = format_predictions(predictions, duration)

    return emotions
```

### 3. Restart Backend

```bash
# Stop the server (Ctrl+C)
# Restart
python -m app.main
```

The frontend requires **no changes** - it will automatically use the real model outputs!

---

## Troubleshooting

### Backend Issues

#### Port 8000 Already in Use

```bash
# Option 1: Kill the process using port 8000
# On macOS/Linux:
lsof -ti:8000 | xargs kill -9

# On Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Option 2: Change the port
# Edit backend/app/main.py, line with uvicorn.run:
uvicorn.run(app, host="0.0.0.0", port=8001)
```

#### Module Not Found Errors

```bash
# Make sure virtual environment is activated
source venv/bin/activate  # or venv\Scripts\activate

# Reinstall dependencies
pip install -r requirements.txt
```

#### "No module named 'app'"

```bash
# Make sure you're in the backend directory
cd SystemCode/backend

# Run with python -m
python -m app.main
```

### Frontend Issues

#### Port 5173 Already in Use

Vite will automatically use the next available port (5174, 5175, etc.)

#### "Cannot find module" Errors

```bash
# Delete node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

#### Tailwind CSS Not Working

```bash
# Reinstall Tailwind
npm uninstall tailwindcss postcss autoprefixer
npm install -D tailwindcss@3 postcss autoprefixer

# Verify index.css has these lines at the top:
@tailwind base;
@tailwind components;
@tailwind utilities;
```

#### CORS Errors in Browser Console

Make sure:

1. Backend is running on port 8000
2. Frontend is running on port 5173
3. `backend/app/main.py` has correct CORS settings:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### General Issues

#### Changes Not Reflecting

- **Backend**: Restart the server (Ctrl+C, then `python -m app.main`)
- **Frontend**: Vite auto-reloads, but try hard refresh (Ctrl+Shift+R)

#### Audio Upload Fails

- Check file format (should be audio file: mp3, wav, m4a, etc.)
- Check browser console (F12) for error messages
- Check backend terminal for error logs

---

## Quick Reference Commands

### Start Backend

```bash
cd SystemCode/backend
source venv/bin/activate  # or venv\Scripts\activate on Windows
python -m app.main
```

### Start Frontend

```bash
cd SystemCode/frontend
npm run dev
```

### Stop Servers

Press `Ctrl+C` in each terminal

### Access Application

- **Frontend UI**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs (auto-generated by FastAPI)

---

## Development Workflow

1. **Make changes** to code
2. **Backend**: Restart server to see changes
3. **Frontend**: Changes auto-reload (Vite hot module replacement)
4. **Test** in browser at http://localhost:5173
5. **Check errors** in browser console (F12) and terminal outputs

---

## Next Steps

- [ ] Implement real emotion detection models
- [ ] Add model training pipeline
- [ ] Add audio playback controls
- [ ] Add export results functionality
- [ ] Add user authentication
- [ ] Deploy to production

---

## Support

For issues or questions:

1. Check the [Troubleshooting](#troubleshooting) section
2. Check browser console for errors (F12)
3. Check terminal outputs for backend errors
4. Review FastAPI auto-docs at http://localhost:8000/docs

---

**Happy Coding! 🎉**
