from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import butter, sosfilt


AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


@dataclass(frozen=True)
class FeatureConfig:
    target_sr: int = 16000
    max_duration: float = 5.0
    n_mfcc: int = 20
    trim_silence: bool = True
    normalize_volume: bool = True
    telephone_bandpass: bool = False


def discover_audio_files(dataset_root: Path, split: str) -> list[tuple[Path, str, str]]:
    """Return (path, label, speaker_id) rows for a split/label directory dataset."""
    split_dir = dataset_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Dataset split not found: {split_dir}")

    rows: list[tuple[Path, str, str]] = []
    for label_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        for audio_path in sorted(label_dir.iterdir()):
            if audio_path.suffix.lower() in AUDIO_EXTENSIONS:
                rows.append((audio_path, label_dir.name, parse_speaker_id(audio_path)))

    if not rows:
        raise FileNotFoundError(f"No audio files found under {split_dir}")
    return rows


def parse_speaker_id(audio_path: Path) -> str:
    """CREMA-D filenames start with ActorID, e.g. 1001_IEO_HAP_HI.wav."""
    return audio_path.stem.split("_")[0]


def load_audio(path: Path, config: FeatureConfig) -> tuple[np.ndarray, int]:
    audio, sr = librosa.load(path, sr=config.target_sr, mono=True)

    if config.trim_silence and audio.size:
        audio, _ = librosa.effects.trim(audio, top_db=25)

    max_samples = int(config.max_duration * sr)
    if max_samples > 0 and audio.size > max_samples:
        audio = audio[:max_samples]

    if config.telephone_bandpass and audio.size:
        audio = apply_telephone_bandpass(audio, sr)

    if config.normalize_volume and audio.size:
        rms = np.sqrt(np.mean(np.square(audio)))
        if rms > 1e-8:
            audio = audio / rms * 0.1
            audio = np.clip(audio, -1.0, 1.0)

    return audio.astype(np.float32), sr


def apply_telephone_bandpass(audio: np.ndarray, sr: int) -> np.ndarray:
    """Approximate telephone channel bandwidth, useful for phone-recording robustness."""
    low = 300 / (sr / 2)
    high = min(3400 / (sr / 2), 0.99)
    sos = butter(4, [low, high], btype="bandpass", output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def extract_features(audio: np.ndarray, sr: int, config: FeatureConfig) -> np.ndarray:
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


def summarize_frames(values: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            np.mean(values, axis=1),
            np.std(values, axis=1),
            np.min(values, axis=1),
            np.max(values, axis=1),
        ]
    ).astype(np.float32)


def extract_file_features(path: Path, config: FeatureConfig) -> np.ndarray:
    audio, sr = load_audio(path, config)
    return extract_features(audio, sr, config)
