import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import os

class EmotionClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(EmotionClassifier, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Global model variable
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _load_model():
    global model
    if model is None:
        model = EmotionClassifier()
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'training', 'model_resnet50.pth')

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            model.resnet.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            print(f"Model loaded from {model_path}")
        else:
            print(f"Model file not found at {model_path}. Using dummy predictions.")
            model = None


def load_model(model_path: str = None):
    global model
    if model is None:
        _load_model()
    return model

# Load model at import time
_load_model()

def predict_emotion(spectrogram):
    """
    Predict emotion from spectrogram
    spectrogram: numpy array of shape (128, time_steps) or similar
    """
    if model is None:
        emotions = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']
        import numpy as np
        probs = np.random.dirichlet(np.ones(6))
        return {emotion: float(prob) for emotion, prob in zip(emotions, probs)}

    import numpy as np

    spec = np.array(spectrogram, dtype=np.float32)
    if spec.std() > 0:
        spec = (spec - spec.mean()) / (spec.std() + 1e-9)
    else:
        spec = spec - spec.mean()

    if spec.ndim == 2:
        spec = np.stack([spec] * 3, axis=0)

    spec_tensor = torch.from_numpy(spec).unsqueeze(0).to(device)
    spec_tensor = F.interpolate(spec_tensor, size=(224, 224), mode='bilinear', align_corners=False)

    with torch.no_grad():
        outputs = model(spec_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    emotions = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']
    return {emotion: float(prob) for emotion, prob in zip(emotions, probabilities)}
