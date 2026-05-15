# Frontend — Audio Emotion Analyzer UI

React + Vite interface to upload or record audio, choose models, and view emotion predictions with spectrograms.

## Run

From the repo root run `./start.sh` (on Windows, inside WSL). It runs `npm install` when needed and starts the Vite dev server. The backend must also be running — the start script launches both.

Frontend: http://localhost:5173. It calls the backend at `http://localhost:8000/api` (set in `src/utils/api.js`).

## Structure

```
frontend/src/
  components/
    AudioEmotionAnalyzer.js   main component; owns audio/loading/results/model state
    AudioInput.js             file upload + microphone recording
    ModelSelection.js         model chooser
    EmotionResults.js         top emotion + distribution
    SpectrogramResults.js     spectrogram + per-model results
  utils/api.js                backend calls
  App.js / main.js            root component + entry point
  index.css                   Tailwind entry
```

## API calls (`src/utils/api.js`)

- `fetchAvailableModels()` → `GET /api/models` (falls back to a default model list if the backend is unreachable).
- `analyzeAudio(file, selectedModels)` → `POST /api/analyze` for multi-model analysis.
- `analyzeEmotion(file, modelId)` → `POST /api/analyze-emotion` for a single model.

To target a different backend, edit `API_BASE_URL` in `src/utils/api.js`.

## Build

`npm run build` (output in `dist/`); preview with `npm run preview`.
