from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from mfcc_features import FeatureConfig, extract_file_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict emotion for one audio file with a trained MFCC model.")
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--model-path", type=Path, default=Path("model_mfcc_svm.joblib"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    model_path = args.model_path if args.model_path.is_absolute() else script_dir / args.model_path
    audio_path = args.audio_path if args.audio_path.is_absolute() else Path.cwd() / args.audio_path

    bundle = joblib.load(model_path)
    config = FeatureConfig(**bundle["feature_config"])
    model = bundle["model"]
    labels = bundle["labels"]

    features = extract_file_features(audio_path, config).reshape(1, -1)
    prediction = model.predict(features)[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        scores = {label: float(prob) for label, prob in zip(model.classes_, probabilities)}
    else:
        scores = {label: 1.0 if label == prediction else 0.0 for label in labels}

    result = {
        "audio": str(audio_path),
        "prediction": prediction,
        "scores": dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
