import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from app.models.predictors.device import select_torch_device

EMOTION_LABELS = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
IMAGE_SIZE = 224

_DEVICE = select_torch_device()

_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class _EmotionClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = models.resnet50(weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


class ResNetPredictor:
    def __init__(self, weight_path):
        self._weight_path = weight_path
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        if not os.path.exists(self._weight_path):
            raise FileNotFoundError(self._weight_path)
        checkpoint = torch.load(self._weight_path, map_location=_DEVICE)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        model = _EmotionClassifier(len(EMOTION_LABELS))
        model.resnet.load_state_dict(state_dict)
        model.to(_DEVICE)
        model.eval()
        self._model = model

    def predict(self, image):
        self._load()
        tensor = _TRANSFORM(image).unsqueeze(0).to(_DEVICE)
        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return {label: float(p) for label, p in zip(EMOTION_LABELS, probs)}
