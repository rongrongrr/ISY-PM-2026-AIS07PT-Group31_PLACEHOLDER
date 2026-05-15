from typing import Protocol
from PIL import Image


class Predictor(Protocol):
    def predict(self, image: Image.Image) -> dict[str, float]:
        ...
