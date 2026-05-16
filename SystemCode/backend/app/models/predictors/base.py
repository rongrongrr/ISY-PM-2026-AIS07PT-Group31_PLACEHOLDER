from typing import Protocol
from PIL import Image
import numpy as np


class Predictor(Protocol):
    def predict(
        self,
        image: Image.Image,
        audio_data: np.ndarray | None = None,
        sr: int | None = None,
    ) -> dict[str, float]:
        ...
