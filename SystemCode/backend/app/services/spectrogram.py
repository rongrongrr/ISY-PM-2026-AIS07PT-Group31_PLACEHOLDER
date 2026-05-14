import librosa
import numpy as np

def generate_spectrogram(audio_data, sr, n_mels=128, n_fft=2048, hop_length=512):
    """
    Generate a normalized mel spectrogram matrix for frontend visualization.
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-10)
    return normalized.astype(np.float32)


def generate_spectrogram_for_model(audio_data, sr, n_mels=128, n_fft=2048, hop_length=512, max_time_steps=500):
    """
    Generate a normalized mel spectrogram matrix for model input.
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-10)

    if normalized.shape[1] < max_time_steps:
        pad_width = max_time_steps - normalized.shape[1]
        normalized = np.pad(normalized, ((0, 0), (0, pad_width)), mode="constant")
    else:
        normalized = normalized[:, :max_time_steps]

    return normalized.astype(np.float32)
