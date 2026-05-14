import io
import os
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio


def process_audio(audio_bytes: bytes, target_sr: int = 16000, filename: str | None = None):
    """
    Process audio bytes and return audio array and sample rate.

    This function first attempts to load directly from in-memory bytes. If that
    fails, it writes the bytes to a temporary file and loads using torchaudio or
    soundfile.
    """
    audio_buffer = io.BytesIO(audio_bytes)
    if filename:
        audio_buffer.name = filename

    try:
        audio_data, sr = librosa.load(audio_buffer, sr=target_sr, mono=True)
        return audio_data, sr
    except Exception:
        return _load_audio_fallback(audio_bytes, target_sr, filename)


def _load_audio_fallback(audio_bytes: bytes, target_sr: int, filename: str | None):
    suffix = Path(filename).suffix if filename else ".wav"
    if not suffix:
        suffix = ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio.flush()
        temp_path = temp_audio.name

    try:
        waveform = None
        sr = None

        try:
            waveform, sr = torchaudio.load(temp_path)
        except Exception:
            try:
                import soundfile as sf
                data, sr = sf.read(temp_path, dtype="float32")
                if data.ndim > 1:
                    data = data.mean(axis=1)
                waveform = torch.from_numpy(data)
            except Exception:
                return _load_audio_via_ffmpeg(temp_path, target_sr)

        if waveform is None:
            return _load_audio_via_ffmpeg(temp_path, target_sr)

        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)

        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform.unsqueeze(0), sr, target_sr).squeeze(0)
            sr = target_sr

        audio_data = waveform.cpu().numpy().astype(np.float32)
        return audio_data, sr
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def _load_audio_via_ffmpeg(temp_path: str, target_sr: int):
    output_path = f"{temp_path}.wav"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        temp_path,
        "-ar",
        str(target_sr),
        "-ac",
        "1",
        output_path,
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
        audio_data, sr = librosa.load(output_path, sr=target_sr, mono=True)
        return audio_data, sr
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg is not installed or not available in PATH. "
            "Please install ffmpeg or record audio in WAV format in the browser."
        ) from exc
    finally:
        for path in (temp_path, output_path):
            try:
                os.remove(path)
            except OSError:
                pass
