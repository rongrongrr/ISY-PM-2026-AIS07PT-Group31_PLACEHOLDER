import os
from ultralytics import YOLO
from app.models.predictors.device import select_torch_device

EMOTION_LABELS = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]


class YoloPredictor:
    def __init__(self, weight_path):
        self._weight_path = weight_path
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        if not os.path.exists(self._weight_path):
            raise FileNotFoundError(self._weight_path)
        self._model = YOLO(self._weight_path)

    def predict(self, image, audio_data=None, sr=None):
        self._load()
        device = select_torch_device()
        ultra_device = "cpu" if device.type == "cpu" else 0
        result = self._model.predict(image, verbose=False, device=ultra_device)[0]
        names = result.names
        probs = result.probs.data.tolist()
        scored = {names[i]: float(p) for i, p in enumerate(probs)}
        return {label: scored.get(label, 0.0) for label in EMOTION_LABELS}
