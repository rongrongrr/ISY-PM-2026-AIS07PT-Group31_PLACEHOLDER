import librosa
import numpy as np

def generate_spectrogram(audio_data, sr, n_mels=128, n_fft=2048):
    """
    Generate mel spectrogram from audio data
    """
    # Generate mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data, 
        sr=sr, 
        n_mels=n_mels,
        n_fft=n_fft
    )
    
    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize for visualization (0-100 range for frontend)
    normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
    spectrogram_data = (normalized * 100).astype(np.float32)
    
    # Average across frequency bins to get simplified data for frontend
    # Returns array of 100 values representing time slices
    time_slices = np.linspace(0, spectrogram_data.shape[1], 100, dtype=int)
    simplified = np.mean(spectrogram_data[:, time_slices], axis=0)
    
    return simplified