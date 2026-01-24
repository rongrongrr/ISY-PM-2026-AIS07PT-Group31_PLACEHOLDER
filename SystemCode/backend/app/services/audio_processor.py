import librosa
import numpy as np
import io

def process_audio(audio_bytes: bytes, target_sr: int = 16000):
    """
    Process audio bytes and return audio array and sample rate
    """
    audio_buffer = io.BytesIO(audio_bytes)
    audio_data, sr = librosa.load(audio_buffer, sr=target_sr)
    return audio_data, sr