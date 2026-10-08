import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_NAME = "diabetes_best_model_pipeline"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT


def save_validation_results(
    output_dir: Path,
    model_name: str,
    metrics: dict[str, Any],
    dataset_shape: tuple[int, int],
    prediction_count: int,
) -> Path:
    result_dir = output_dir / "results" / "experiments" / Path(model_name).name
    result_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = result_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report_lines = [
        f"Model: {Path(model_name).name}",
        f"Dataset shape: {dataset_shape}",
        f"Predictions: {prediction_count}",
    ]
    report_lines.extend(f"{name}: {value:.4f}" for name, value in metrics.items())
    report_path = result_dir / "validation_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return result_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a saved diabetes pipeline")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    data_path = output_dir / "data" / "processed" / "diabetes.csv"
    model_path = output_dir / "models" / f"{Path(args.model_name).name}.joblib"

    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    test_df = pd.read_csv(data_path)
    X_test = test_df.drop(columns=["Outcome"])
    y_test = test_df["Outcome"]

    pipeline = joblib.load(model_path)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    print(f"Dataset: {test_df.shape}")
    print(f"Model: {model_path}")
    print(f"Predictions: {len(predictions)}")
    print("Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    result_dir = save_validation_results(
        output_dir=output_dir,
        model_name=Path(args.model_name).name,
        metrics=metrics,
        dataset_shape=tuple(test_df.shape),
        prediction_count=len(predictions),
    )
    print(f"Results: {result_dir}")


if __name__ == "__main__":
    main()
