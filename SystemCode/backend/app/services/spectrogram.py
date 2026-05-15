import io
import librosa
import numpy as np
from PIL import Image
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

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


MODEL_SR = 16000
MODEL_N_FFT = 2048
MODEL_HOP_LENGTH = 64
MODEL_N_MELS = 512
MODEL_TOP_DB = 20
MODEL_IMG_PX = 640
MODEL_DPI = 300


def generate_model_input_image(audio_data, sr):
    y = np.asarray(audio_data, dtype=np.float32)

    if sr != MODEL_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=MODEL_SR)
        sr = MODEL_SR

    trimmed, _ = librosa.effects.trim(y, top_db=MODEL_TOP_DB)
    if trimmed.size > 0:
        y = trimmed

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=MODEL_N_FFT,
        hop_length=MODEL_HOP_LENGTH,
        n_mels=MODEL_N_MELS,
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    fig = Figure(figsize=(MODEL_IMG_PX / MODEL_DPI, MODEL_IMG_PX / MODEL_DPI), dpi=MODEL_DPI)
    FigureCanvasAgg(fig)
    ax = fig.subplots(1, 1)
    ax.imshow(S_db, aspect="auto", origin="lower", cmap="gray")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=MODEL_DPI, bbox_inches="tight", pad_inches=0)
    fig.clf()
    buf.seek(0)
    image = Image.open(buf).convert("RGB")
    buf.close()
    return image
