from pathlib import Path
from app.models.predictors.resnet import ResNetPredictor
from app.models.predictors.yolo import YoloPredictor

_MODELS_DIR = Path(__file__).resolve().parents[4] / "data" / "training"

DEFAULT_MODEL_ID = "model_resnet50.pth"

_FACTORIES = {
    "model_resnet50.pth": lambda: ResNetPredictor(str(_MODELS_DIR / "model_resnet50.pth")),
    "model_yolo.pt": lambda: YoloPredictor(str(_MODELS_DIR / "model_yolo.pt")),
}

_CACHE = {}


def available_model_ids():
    return list(_FACTORIES.keys())


def get_predictor(model_id):
    if model_id not in _FACTORIES:
        raise KeyError(model_id)
    if model_id not in _CACHE:
        _CACHE[model_id] = _FACTORIES[model_id]()
    return _CACHE[model_id]
