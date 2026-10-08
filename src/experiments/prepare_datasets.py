import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from model_pipeline import create_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT
DEFAULT_MODEL_NAME = "diabetes_best_model_pipeline"


def load_and_prepare_data(
    data_path: Path, processed_data_path: Path
) -> tuple[pd.DataFrame, pd.Series]:
    if data_path.is_dir():
        source_files = sorted(data_path.glob("*.csv"))
    else:
        source_files = [data_path]
    if not source_files:
        raise FileNotFoundError(f"No CSV files found in data path: {data_path}")

    frames = [pd.read_csv(path) for path in source_files]
    data = pd.concat(frames, ignore_index=True).drop_duplicates()
    data = data.sort_values(
        by=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
            "Outcome",
        ]
    ).reset_index(drop=True)

    zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "BMI"]
    for column in zero_as_missing:
        data[column] = data[column].replace(0, pd.NA)
    data = data.dropna(subset=["Glucose"])
    data = data[
        (data["BloodPressure"] >= 20)
        & (data["BloodPressure"] <= 130)
        & (data["BMI"] >= 12)
        & (data["BMI"] <= 70)
        & (data["SkinThickness"] <= 99)
        & (data["Insulin"] <= 800)
    ].reset_index(drop=True)

    processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(processed_data_path, index=False)
    return data.drop(columns=["Outcome"]), data["Outcome"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare diabetes data and train a model"
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be greater than zero")
    if args.random_seed < 0:
        parser.error("--random-seed must be non-negative")

    data_path = args.data_path.resolve()
    output_dir = args.output_dir.resolve()
    processed_data_path = output_dir / "data" / "processed" / "diabetes.csv"
    model_dir = output_dir / "models"
    model_name = Path(args.model_name).name
    if not model_name:
        parser.error("--model-name must not be empty")
    model_path = model_dir / f"{model_name}.joblib"

    X, y = load_and_prepare_data(data_path, processed_data_path)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=args.random_seed
    )

    pipeline = create_pipeline(random_seed=args.random_seed, epochs=args.epochs)
    pipeline.fit(X_train, y_train)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    print(f"Prepared dataset: {processed_data_path}")
    print(f"Saved model: {model_path}")
    print(f"Dataset rows: {len(X)}")


if __name__ == "__main__":
    main()
