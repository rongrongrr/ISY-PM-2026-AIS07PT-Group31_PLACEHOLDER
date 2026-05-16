import os
from dataclasses import dataclass

import joblib
import librosa
import numpy as np
from scipy.signal import butter, sosfilt


EMOTION_LABELS = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]


@dataclass(frozen=True)
class FeatureConfig:
    target_sr: int = 16000
    max_duration: float = 5.0
    n_mfcc: int = 20
    trim_silence: bool = True
    normalize_volume: bool = True
    telephone_bandpass: bool = False


class MfccPredictor:
    def __init__(self, weight_path):
        self._weight_path = weight_path
        self._model = None
        self._labels = EMOTION_LABELS
        self._config = FeatureConfig()

    def _load(self):
        if self._model is not None:
            return
        if not os.path.exists(self._weight_path):
            raise FileNotFoundError(self._weight_path)

        bundle = joblib.load(self._weight_path)
        self._model = bundle["model"]
        self._labels = bundle.get("labels", EMOTION_LABELS)
        self._config = FeatureConfig(**bundle.get("feature_config", {}))

    def predict(self, image, audio_data=None, sr=None):
        self._load()
        if audio_data is None or sr is None:
            raise ValueError("MfccPredictor requires audio_data and sr for inference.")

        audio = prepare_audio(audio_data, sr, self._config)
        features = extract_features(audio, self._config.target_sr, self._config).reshape(1, -1)
        prediction = self._model.predict(features)[0]

        if hasattr(self._model, "predict_proba"):
            probabilities = self._model.predict_proba(features)[0]
            scores = {label: float(prob) for label, prob in zip(self._model.classes_, probabilities)}
        else:
            scores = {label: 1.0 if label == prediction else 0.0 for label in self._labels}

        return {label: scores.get(label, 0.0) for label in EMOTION_LABELS}


def prepare_audio(audio_data, sr, config):
    audio = np.asarray(audio_data, dtype=np.float32)
    if sr != config.target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=config.target_sr)

    if config.trim_silence and audio.size:
        audio, _ = librosa.effects.trim(audio, top_db=25)

    max_samples = int(config.max_duration * config.target_sr)
    if max_samples > 0 and audio.size > max_samples:
        audio = audio[:max_samples]

    if config.telephone_bandpass and audio.size:
        audio = apply_telephone_bandpass(audio, config.target_sr)

    if config.normalize_volume and audio.size:
        rms = np.sqrt(np.mean(np.square(audio)))
        if rms > 1e-8:
            audio = audio / rms * 0.1
            audio = np.clip(audio, -1.0, 1.0)

    return audio.astype(np.float32)


def apply_telephone_bandpass(audio, sr):
    low = 300 / (sr / 2)
    high = min(3400 / (sr / 2), 0.99)
    sos = butter(4, [low, high], btype="bandpass", output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def extract_features(audio, sr, config):
    if audio.size == 0:
        audio = np.zeros(int(0.1 * sr), dtype=np.float32)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=config.n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
    rms = librosa.feature.rms(y=audio)

    blocks = [
        summarize_frames(mfcc),
        summarize_frames(delta),
        summarize_frames(delta2),
        summarize_frames(chroma),
        summarize_frames(spectral_centroid),
        summarize_frames(spectral_bandwidth),
        summarize_frames(spectral_rolloff),
        summarize_frames(zero_crossing_rate),
        summarize_frames(rms),
        np.array([len(audio) / sr], dtype=np.float32),
    ]
    return np.concatenate(blocks).astype(np.float32)


def summarize_frames(values):
    return np.concatenate(
        [
            np.mean(values, axis=1),
            np.std(values, axis=1),
            np.min(values, axis=1),
            np.max(values, axis=1),
        ]
    ).astype(np.float32)
