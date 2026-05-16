from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
from mfcc_features import FeatureConfig, discover_audio_files, extract_file_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a classical ML emotion classifier using MFCC-style audio features."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("../../../datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--model-name", default="model_mfcc_svm.joblib")
    parser.add_argument("--classifier", choices=["svm", "random_forest"], default="svm")
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--max-duration", type=float, default=5.0)
    parser.add_argument("--n-mfcc", type=int, default=20)
    parser.add_argument("--no-trim-silence", action="store_true")
    parser.add_argument("--no-normalize-volume", action="store_true")
    parser.add_argument("--telephone-bandpass", action="store_true")
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--cv", action="store_true", help="Run a small grouped grid search before final training.")
    parser.add_argument("--force", action="store_true", help="Recompute cached feature files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    dataset_root = resolve_path(args.dataset_root, script_dir)
    output_dir = resolve_path(args.output_dir, script_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = FeatureConfig(
        target_sr=args.target_sr,
        max_duration=args.max_duration,
        n_mfcc=args.n_mfcc,
        trim_silence=not args.no_trim_silence,
        normalize_volume=not args.no_normalize_volume,
        telephone_bandpass=args.telephone_bandpass,
    )

    print(f"Dataset root: {dataset_root}")
    print(f"Output dir:   {output_dir}")
    train_rows = maybe_limit_rows(discover_audio_files(dataset_root, "train"), args.limit_per_class)
    test_rows = maybe_limit_rows(discover_audio_files(dataset_root, "test"), args.limit_per_class)

    cache_key = make_cache_key(config, args.limit_per_class)
    x_train_all, y_train_all, groups_train = build_feature_matrix(
        train_rows,
        config,
        output_dir / f"mfcc_train_{cache_key}.npz",
        force=args.force,
    )
    x_test, y_test, _ = build_feature_matrix(
        test_rows,
        config,
        output_dir / f"mfcc_test_{cache_key}.npz",
        force=args.force,
    )

    x_train, x_val, y_train, y_val, groups_fit, _ = split_train_validation(
        x_train_all,
        y_train_all,
        groups_train,
        validation_size=args.validation_size,
    )

    estimator = build_estimator(args.classifier)
    if args.cv:
        estimator = tune_estimator(estimator, x_train, y_train, groups_fit, args.classifier)

    estimator.fit(x_train, y_train)

    metrics = {
        "validation": evaluate_split(estimator, x_val, y_val),
        "test": evaluate_split(estimator, x_test, y_test),
        "classes": sorted(np.unique(y_train_all).tolist()),
        "train_samples": int(len(y_train)),
        "validation_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
        "feature_config": asdict(config),
        "classifier": args.classifier,
    }

    model_path = output_dir / args.model_name
    joblib.dump(
        {
            "model": estimator,
            "labels": metrics["classes"],
            "feature_config": asdict(config),
            "metadata": metrics,
        },
        model_path,
    )

    write_reports(output_dir, estimator, x_test, y_test, metrics)
    print(json.dumps(metrics, indent=2))
    print(f"Saved model: {model_path}")


def resolve_path(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else (base / path).resolve()


def make_cache_key(config: FeatureConfig, limit_per_class: int | None) -> str:
    limit = "all" if limit_per_class is None else f"limit{limit_per_class}"
    phone = "phone" if config.telephone_bandpass else "clean"
    trim = "trim" if config.trim_silence else "notrim"
    norm = "norm" if config.normalize_volume else "nonorm"
    return f"sr{config.target_sr}_dur{config.max_duration:g}_mfcc{config.n_mfcc}_{phone}_{trim}_{norm}_{limit}"


def maybe_limit_rows(rows: list[tuple[Path, str, str]], limit_per_class: int | None) -> list[tuple[Path, str, str]]:
    if limit_per_class is None:
        return rows

    counts: dict[str, int] = {}
    limited: list[tuple[Path, str, str]] = []
    for row in rows:
        label = row[1]
        if counts.get(label, 0) >= limit_per_class:
            continue
        limited.append(row)
        counts[label] = counts.get(label, 0) + 1
    return limited


def build_feature_matrix(
    rows: list[tuple[Path, str, str]],
    config: FeatureConfig,
    cache_path: Path,
    force: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cache_path.exists() and not force:
        cached = np.load(cache_path, allow_pickle=True)
        print(f"Loaded cached features: {cache_path}")
        return cached["x"], cached["y"], cached["groups"]

    x_values: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[str] = []
    total = len(rows)
    for index, (audio_path, label, speaker_id) in enumerate(rows, start=1):
        x_values.append(extract_file_features(audio_path, config))
        labels.append(label)
        groups.append(speaker_id)
        if index == 1 or index % 100 == 0 or index == total:
            print(f"Extracted {index}/{total}: {audio_path.name}")

    x = np.vstack(x_values)
    y = np.array(labels)
    group_array = np.array(groups)
    np.savez_compressed(cache_path, x=x, y=y, groups=group_array)
    print(f"Saved feature cache: {cache_path}")
    return x, y, group_array


def split_train_validation(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    validation_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unique_groups = np.unique(groups)
    if len(unique_groups) >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=validation_size, random_state=42)
        train_idx, val_idx = next(splitter.split(x, y, groups))
    else:
        train_idx, val_idx = train_test_split(
            np.arange(len(y)),
            test_size=validation_size,
            random_state=42,
            stratify=y,
        )

    return x[train_idx], x[val_idx], y[train_idx], y[val_idx], groups[train_idx], groups[val_idx]


def build_estimator(classifier: str) -> Pipeline:
    if classifier == "svm":
        model = SVC(C=10.0, gamma="scale", kernel="rbf", class_weight="balanced", probability=True)
        return Pipeline([("scaler", StandardScaler()), ("classifier", model)])

    model = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("classifier", model)])


def tune_estimator(estimator: Pipeline, x: np.ndarray, y: np.ndarray, groups: np.ndarray, classifier: str) -> Pipeline:
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        print("Skipping CV because fewer than two speaker groups are available.")
        return estimator

    if classifier == "svm":
        params = {
            "classifier__C": [1.0, 10.0, 30.0],
            "classifier__gamma": ["scale", 0.01, 0.001],
        }
    else:
        params = {
            "classifier__max_depth": [None, 20, 40],
            "classifier__min_samples_leaf": [1, 2, 4],
        }

    search = GridSearchCV(
        estimator,
        params,
        cv=GroupKFold(n_splits=n_splits),
        scoring="f1_macro",
        n_jobs=-1,
        verbose=2,
    )
    search.fit(x, y, groups=groups)
    print(f"Best CV params: {search.best_params_}")
    print(f"Best CV macro F1: {search.best_score_:.4f}")
    return search.best_estimator_


def evaluate_split(estimator: Pipeline, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    predictions = estimator.predict(x)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro")),
    }


def write_reports(output_dir: Path, estimator: Pipeline, x_test: np.ndarray, y_test: np.ndarray, metrics: dict) -> None:
    predictions = estimator.predict(x_test)
    labels = metrics["classes"]

    metrics_path = output_dir / "mfcc_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report_path = output_dir / "mfcc_classification_report.txt"
    report_path.write_text(classification_report(y_test, predictions, labels=labels), encoding="utf-8")

    matrix = confusion_matrix(y_test, predictions, labels=labels)
    matrix_path = output_dir / "mfcc_confusion_matrix.csv"
    with matrix_path.open("w", encoding="utf-8") as file:
        file.write("," + ",".join(labels) + "\n")
        for label, row in zip(labels, matrix):
            file.write(label + "," + ",".join(str(int(value)) for value in row) + "\n")


if __name__ == "__main__":
    main()
