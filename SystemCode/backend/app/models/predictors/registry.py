from pathlib import Path

_MODELS_DIR = Path(__file__).resolve().parents[4] / "data" / "training"

DEFAULT_MODEL_ID = "model_resnet50.pth"


def _create_resnet_predictor():
    from app.models.predictors.resnet import ResNetPredictor

    return ResNetPredictor(str(_MODELS_DIR / "model_resnet50.pth"))


def _create_yolo_predictor():
    from app.models.predictors.yolo import YoloPredictor

    return YoloPredictor(str(_MODELS_DIR / "model_yolo.pt"))


def _create_mfcc_predictor():
    from app.models.predictors.mfcc import MfccPredictor

    return MfccPredictor(str(_MODELS_DIR / "ml" / "model_mfcc_svm.joblib"))


_FACTORIES = {
    "model_resnet50.pth": _create_resnet_predictor,
    "model_yolo.pt": _create_yolo_predictor,
    "model_mfcc_svm.joblib": _create_mfcc_predictor,
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
