import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier


class ZeroToNanTransformer:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for column in ("Glucose", "BloodPressure", "SkinThickness", "BMI"):
            if column in X.columns:
                X[column] = X[column].replace(0, np.nan)
        return X


class NotebookFeatureEngineer:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["BMI_Category"] = pd.cut(
            X["BMI"], bins=[-np.inf, 18.5, 25, 30, np.inf], labels=[0, 1, 2, 3]
        ).cat.codes
        X["Age_Group"] = pd.cut(
            X["Age"], bins=[-np.inf, 30, 40, 50, np.inf], labels=[0, 1, 2, 3]
        ).cat.codes
        X["Glucose_High"] = (X["Glucose"] >= 140).astype(int)
        X["Glucose_BMI"] = X["Glucose"] * X["BMI"]
        X["Insulin_Glucose_Ratio"] = X["Insulin"] / (X["Glucose"] + 1e-6)
        X["Pregnancies_per_Age"] = X["Pregnancies"] / (X["Age"] + 1e-6)
        return X


def create_pipeline():
    return Pipeline(
        steps=[
            ("zero_to_nan", ZeroToNanTransformer()),
            ("feature_engineer", NotebookFeatureEngineer()),
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler()),
            ("model", LGBMClassifier(
                learning_rate=0.05,
                max_depth=3,
                n_estimators=100,
                random_state=42,
                verbose=-1,
            )),
        ]
    )
