# Frontend - Audio Emotion Analyzer UI

React + Vite-based user interface for audio emotion classification. Upload or record audio, then visualize emotion predictions with spectrograms.

## Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the Development Server](#running-the-development-server)
- [How It Works](#how-it-works)
- [Component Architecture](#component-architecture)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# 1. Navigate to frontend
cd SystemCode/frontend

# 2. Install dependencies
npm install

# 3. Start development server
npm run dev
```

✅ Frontend running at: **http://localhost:5173**

**Note:** Backend must be running separately on port 8000.

---

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── AudioEmotionAnalyzer.js      ← Main component
│   │   ├── AudioInput.js                ← Upload/Record UI
│   │   ├── EmotionResults.js            ← Results display
│   │   ├── ModelSelection.js            ← Model chooser
│   │   └── SpectrogramResults.js        ← Spectrogram visualization
│   ├── utils/
│   │   └── api.js                       ← Backend API calls
│   ├── assets/                          ← Static assets
│   ├── App.js                           ← Root component
│   ├── App.css                          ← Global styles
│   ├── main.js                          ← Entry point
│   └── index.css                        ← Tailwind imports
├── public/                              ← Static files
├── package.json                         ← Dependencies
├── vite.config.js                       ← Vite configuration
├── tailwind.config.js                   ← Tailwind CSS config
├── postcss.config.js                    ← PostCSS config
├── eslint.config.js                     ← ESLint config
├── index.html                           ← HTML entry point
└── README.md                            ← This file
```

---

## Setup & Installation

### Prerequisites

- Node.js 16+ (recommended: v18 or higher)
- npm 8+
- Backend running on http://localhost:8000

### Check Versions

```bash
node --version
npm --version
```

### 1. Install Dependencies

```bash
npm install
```

**Installed packages:**

- `react` - UI framework
- `vite` - Build tool & dev server
- `tailwindcss` - Styling
- `lucide-react` - Icons
- `plotly.js-react` - Charts
- Other utilities

### 2. Verify Installation

```bash
npm list react vite tailwindcss
```

You should see versions for each package.

---

## Running the Development Server

### Start Server

```bash
npm run dev
```

**Expected Output:**

```
  VITE v7.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
  ➜  press h + enter to show help
```

### Access Application

Open browser to: **http://localhost:5173**

### Stop Server

Press `Ctrl+C` in the terminal.

### Hot Module Replacement (HMR)

Changes to `.js` and `.css` files automatically reload in the browser. No manual refresh needed!

---

## How It Works

### User Workflow

1. **Load Audio**
   - Click "Upload Audio File" or "Record Audio"
   - Upload a file or record via microphone

2. **Analyze**
   - Click "Analyze Emotions"
   - Frontend sends audio to backend API

3. **Backend Processing**
   - Backend generates mel-spectrogram
   - Runs ResNet50 emotion classifier
   - Returns emotion probabilities

4. **Display Results**
   - Shows top emotion with confidence
   - Displays emotion probabilities (bar chart)
   - Renders spectrogram visualization

### Data Flow

```
┌─────────────────────────────┐
│   User Uploads/Records      │
│         Audio               │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  AudioEmotionAnalyzer.js    │
│  (Main Component)           │
└──────────────┬──────────────┘
               │
        ┌──────┴──────┐
        │             │
        ▼             ▼
    AudioInput   EmotionResults
    (Upload)     (Display)
        │             ▲
        │             │
        └──────┬──────┘
               │
        POST /api/analyze-emotion
               │
               ▼
    ┌──────────────────────┐
    │   Backend (FastAPI)  │
    │   - Load audio       │
    │   - Generate spec    │
    │   - Run model        │
    └──────────────────────┘
```

---

## Component Architecture

### AudioEmotionAnalyzer (Main)

**File:** `src/components/AudioEmotionAnalyzer.js`

Root component. Manages:

- Audio state (file or recorded blob)
- Loading state
- Results state
- Model selection

### AudioInput

**File:** `src/components/AudioInput.js`

Child of AudioEmotionAnalyzer. Provides:

- File upload input
- Microphone recording button
- Recording controls

**Props:**

```javascript
<AudioInput
  onAudioSelect={(blob) => {}}
  onRecordingStart={() => {}}
  onRecordingStop={() => {}}
  isLoading={false}
/>
```

### EmotionResults

**File:** `src/components/EmotionResults.js`

Displays:

- Top emotion card (emotion + confidence %)
- Classification report

**Props:**

```javascript
<EmotionResults
  result={{
    emotions: {ANG: 0.05, DIS: 0.08, ...},
    top_emotion: "HAP",
    confidence: 0.85
  }}
/>
```

### SpectrogramResults

**File:** `src/components/SpectrogramResults.js`

Displays:

- Spectrogram image
- Emotion probabilities (bar chart)
- Visualization controls

**Props:**

```javascript
<SpectrogramResults
  spectrogram="base64_image_data"
  emotions={{ANG: 0.05, DIS: 0.08, ...}}
/>
```

### ModelSelection

**File:** `src/components/ModelSelection.js`

UI for choosing which models to use. (Currently supports dummy model for development)

---

## API Integration

### Backend Connection

The frontend expects the backend at: **http://localhost:8000**

**File:** `src/utils/api.js`

```javascript
const API_BASE_URL = "http://localhost:8000/api";

export const analyzeEmotion = async (audioBlob) => {
  const formData = new FormData();
  formData.append("audio", audioBlob);

  const response = await fetch(`${API_BASE_URL}/analyze-emotion`, {
    method: "POST",
    body: formData,
  });
  return response.json();
};
```

### Changing Backend URL

If backend is on different host/port:

**Edit:** `src/utils/api.js`

```javascript
const API_BASE_URL = "http://YOUR_HOST:YOUR_PORT/api";
```

Then restart frontend: `npm run dev`

---

## Customization

### Change Styling

Edit `src/index.css` for global styles or component-specific CSS in component files.

Tailwind CSS classes are available throughout. Edit `tailwind.config.js` for custom configuration.

### Change Colors

**File:** `tailwind.config.js`

```javascript
module.exports = {
  theme: {
    extend: {
      colors: {
        primary: "#your-color",
        secondary: "#your-color",
      },
    },
  },
};
```

### Add New Components

Create new `.js` file in `src/components/`:

```javascript
export default function MyComponent({ prop1, prop2 }) {
  return (
    <div className="bg-white rounded-lg shadow-md p-4">
      {/* Component content */}
    </div>
  );
}
```

Then import in `AudioEmotionAnalyzer.js`:

```javascript
import MyComponent from "./MyComponent";
```

---

## Troubleshooting

### Backend Not Responding

**Error:** `Failed to fetch` in browser console or `CORS error`

**Solutions:**

1. Verify backend is running: `python -m app.main` in another terminal
2. Verify backend is on port 8000
3. Check browser console (F12) for CORS errors
4. Verify API URL in `src/utils/api.js` is correct

**Debug:**

```bash
# Test backend directly
curl http://localhost:8000
# Should return: {"message": "Audio Emotion Analyzer API"}
```

---

### Port 5173 Already in Use

**Error:** `EADDRINUSE` or port in use error

**Solution:** Vite automatically tries ports 5174, 5175, etc. Check the output for the actual port being used.

Or kill the process:

macOS/Linux:

```bash
lsof -ti:5173 | xargs kill -9
```

Windows:

```bash
netstat -ano | findstr :5173
taskkill /PID <PID> /F
```

---

### Dependencies Not Installing

**Error:** `npm ERR! code ERESOLVE` or dependency conflicts

**Solution:**

```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and lock file
rm -rf node_modules package-lock.json

# Reinstall
npm install
```

---

### Tailwind CSS Not Working

**Error:** Styles not applying, no colors showing

**Solution:**

1. Verify `src/index.css` has Tailwind imports at the top:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

2. Restart dev server: `npm run dev`

3. Check `tailwind.config.js` includes content paths:

```javascript
content: [
  "./index.html",
  "./src/**/*.{js,jsx}",
],
```

---

### "Cannot find module" Errors

**Error:** `Module not found` in browser console

**Solution:**

```bash
# Check that file exists
ls src/components/ComponentName.js

# Verify import path is correct (case-sensitive on Linux/Mac)
import Component from "./ComponentName";  // ✅ Correct
import Component from "./componentname";  // ❌ Wrong
```

---

### HMR Not Working

**Error:** Changes not reflecting automatically

**Solution:**

1. Restart dev server: `Ctrl+C`, then `npm run dev`
2. Hard refresh browser: `Ctrl+Shift+R` (Windows/Linux) or `Cmd+Shift+R` (Mac)
3. Check browser console for errors

---

### Audio Upload Issues

**Error:** "Failed to upload audio" or blank results

**Possible causes:**

1. Large file size
2. Unsupported audio format
3. Corrupted audio file

**Solution:**

- Use common formats: MP3, WAV, M4A
- Keep file size < 100MB for testing
- Check browser console (F12) for error messages
- Check backend terminal for error logs

---

### Microphone Recording Issues

**Error:** "Permission denied" or microphone not working

**Solution:**

1. Browser must request microphone permission
2. Allow permission when browser prompts
3. Check browser settings (may have microphone blocked)
4. Some browsers require HTTPS (except localhost)

---

## Quick Reference

| Task                     | Command                   |
| ------------------------ | ------------------------- |
| Install deps             | `npm install`             |
| Start dev server         | `npm run dev`             |
| Stop server              | `Ctrl+C`                  |
| Build for production     | `npm run build`           |
| Preview production build | `npm run preview`         |
| Clear cache              | `npm cache clean --force` |

---

## Build for Production

```bash
npm run build
```

Creates optimized build in `dist/` folder.

Preview:

```bash
npm run preview
```

---

## Next Steps

1. ✅ Run backend: `python -m app.main`
2. ✅ Run frontend: `npm run dev`
3. ✅ Open http://localhost:5173
4. ✅ Upload/record audio and test
5. ✅ Train real model when ready

---

## Support

- **Vite Docs**: https://vitejs.dev
- **React Docs**: https://react.dev
- **Tailwind Docs**: https://tailwindcss.com
- **FastAPI Docs**: http://localhost:8000/docs
