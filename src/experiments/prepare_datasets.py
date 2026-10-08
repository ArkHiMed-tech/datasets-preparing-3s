from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from model_pipeline import create_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "diabetes.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "diabetes_best_model_pipeline.pkl"


def load_and_prepare_data() -> tuple[pd.DataFrame, pd.Series]:
    source_files = [RAW_DATA_DIR / f"diabetes ({number}).csv" for number in range(1, 9)]
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

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(PROCESSED_DATA_PATH, index=False)
    return data.drop(columns=["Outcome"]), data["Outcome"]


def main() -> None:
    X, y = load_and_prepare_data()
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print(f"Prepared dataset: {PROCESSED_DATA_PATH}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Dataset rows: {len(X)}")


if __name__ == "__main__":
    main()
