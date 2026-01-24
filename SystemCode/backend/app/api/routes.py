from fastapi import APIRouter, UploadFile, File, Form
from typing import List
import json
import numpy as np
from app.services.audio_processor import process_audio
from app.services.spectrogram import generate_spectrogram

router = APIRouter()

# ============================================================================
# TOGGLE THIS FLAG TO SWITCH BETWEEN DUMMY DATA AND REAL MODELS
# ============================================================================
USE_DUMMY_MODELS = True  # Set to False when real models are ready
# ============================================================================

@router.post("/analyze")
async def analyze_audio(
    audio: UploadFile = File(...),
    models: str = Form(...)
):
    """
    Analyze audio file and return spectrograms with emotion labels
    """
    # Parse model IDs
    model_ids = json.loads(models)
    
    # Read audio file
    audio_bytes = await audio.read()
    
    # Process audio
    audio_data, sr = process_audio(audio_bytes)
    duration = len(audio_data) / sr
    
    # Process audio and generate spectrograms for each model
    results = []
    for model_id in model_ids:
        if USE_DUMMY_MODELS:
            # DUMMY MODE: Generate fake data
            spectrogram_data = generate_dummy_spectrogram()
            emotions = generate_dummy_emotions(duration)
        else:
            # REAL MODE: Use actual models
            spectrogram_data = generate_spectrogram(audio_data, sr)
            emotions = predict_emotions(audio_data, sr, model_id)
        
        results.append({
            "modelId": model_id,
            "modelName": get_model_name(model_id),
            "spectrogramData": spectrogram_data.tolist() if hasattr(spectrogram_data, 'tolist') else spectrogram_data,
            "emotions": emotions,
            "duration": float(duration)
        })
    
    return {"spectrograms": results}

def get_model_name(model_id: str) -> str:
    names = {
        "baseline": "Baseline Model",
        "advanced": "Advanced Model",
        "ensemble": "Ensemble Model"
    }
    return names.get(model_id, model_id)

def generate_dummy_spectrogram():
    """Generate dummy spectrogram data for testing"""
    return np.random.uniform(40, 100, 100).tolist()

def generate_dummy_emotions(duration: float):
    """Generate dummy emotion predictions for testing"""
    # Create 4-5 random emotion segments
    num_segments = np.random.randint(4, 6)
    emotions_list = ['Happy', 'Sad', 'Angry', 'Neutral', 'Surprised', 'Fear']
    
    emotions = []
    segment_duration = duration / num_segments
    
    for i in range(num_segments):
        start = i * segment_duration
        end = (i + 1) * segment_duration
        emotion = np.random.choice(emotions_list)
        confidence = np.random.uniform(0.7, 0.95)
        
        emotions.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "emotion": emotion,
            "confidence": round(confidence, 2)
        })
    
    return emotions

def predict_emotions(audio_data, sr, model_id):
    """
    TODO: Implement actual emotion prediction with real models
    This is where you'll load your trained models and make predictions
    """
    # Placeholder for real model implementation
    duration = len(audio_data) / sr
    
    # Example structure - replace with actual model predictions
    return [
        {"start": 0, "end": 1, "emotion": "Happy", "confidence": 0.87},
        {"start": 1, "end": 2.5, "emotion": "Neutral", "confidence": 0.72},
    ]