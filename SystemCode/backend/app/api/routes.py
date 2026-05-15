from fastapi import APIRouter, UploadFile, File, Form
from typing import List
import json
import numpy as np
import os
from pathlib import Path
from app.services.audio_processor import process_audio
from app.services.spectrogram import generate_spectrogram, generate_model_input_image
from app.models.predictors.registry import get_predictor, available_model_ids, DEFAULT_MODEL_ID

router = APIRouter()

# Model configuration: maps model filename/id to display name and description
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

EMOTION_LABELS = {
    "ANG": "Angry",
    "DIS": "Disgust",
    "FEA": "Fear",
    "HAP": "Happy",
    "NEU": "Neutral",
    "SAD": "Sad",
}

# ============================================================================
# TOGGLE THIS FLAG TO SWITCH BETWEEN DUMMY DATA AND REAL MODELS
# ============================================================================
USE_DUMMY_MODELS = False  # Set to False when real models are ready
# ============================================================================


def localize_emotion_label(label: str) -> str:
    return EMOTION_LABELS.get(label, label)


@router.get("/models")
async def list_available_models():
    """
    List all available emotion classification models
    """
    try:
        models_dir = Path(__file__).parent.parent.parent.parent / "data" / "training"
        print(f"[DEBUG] Looking for models in: {models_dir}")
        print(f"[DEBUG] Directory exists: {models_dir.exists()}")
        
        available_models = []
        
        # First try to find models in MODEL_CONFIG
        for model_id, config in MODEL_CONFIG.items():
            model_path = models_dir / model_id
            print(f"[DEBUG] Checking {model_id}: exists={model_path.exists()}")
            if model_path.exists():
                available_models.append({
                    "id": config["id"],
                    "name": config["name"],
                    "description": config["description"],
                    "path": str(model_path)
                })
        
        # If no models found in config, try to auto-discover .pth files
        if not available_models and models_dir.exists():
            print("[DEBUG] No configured models found, auto-discovering...")
            for pth_file in models_dir.glob("*.pth"):
                print(f"[DEBUG] Found model file: {pth_file}")
                available_models.append({
                    "id": pth_file.stem,
                    "name": pth_file.stem.replace("_", " ").title(),
                    "description": f"Model: {pth_file.name}",
                    "path": str(pth_file)
                })
        
        print(f"[DEBUG] Returning {len(available_models)} models")
        return {"models": available_models}
    
    except Exception as e:
        print(f"[ERROR] Failed to list models: {e}")
        import traceback
        traceback.print_exc()
        return {"models": [], "error": str(e)}

@router.post("/analyze")
async def analyze_audio(
    audio: UploadFile = File(...),
    models: str = Form(...)
):
    """
    Analyze audio file and return spectrogram and emotion labels
    """
    model_ids = json.loads(models)
    audio_bytes = await audio.read()
    audio_data, sr = process_audio(audio_bytes, filename=audio.filename)
    duration = len(audio_data) / sr

    results = []
    for model_id in model_ids:
        if USE_DUMMY_MODELS:
            spectrogram_data = generate_dummy_spectrogram()
            emotion_probs = {"DUMMY": 1.0}
            top_emotion = "DUMMY"
            confidence = 1.0
            segments = generate_dummy_emotions(duration)
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

        results.append({
            "modelId": model_id,
            "modelName": get_model_name(model_id),
            "duration": float(duration),
            "sampleRate": sr,
            "spectrogramData": spectrogram_data.tolist() if isinstance(spectrogram_data, np.ndarray) else spectrogram_data,
            "emotions": emotion_probs,
            "emotionDistribution": [
                {"emotion": localize_emotion_label(label), "label": label, "confidence": float(prob)}
                for label, prob in emotion_probs.items()
            ],
            "topEmotion": top_emotion,
            "topEmotionName": localize_emotion_label(top_emotion),
            "confidence": confidence,
            "segments": segments,
        })

    return {"results": results}


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

