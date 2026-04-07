from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass(slots=True)
class TrainingResult:
    model_name: str
    accuracy: float
    macro_f1: float
    report: str


def split_features_and_labels(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x_data = frame.drop(columns=["image_path", "label"])
    y_data = frame["label"]
    return x_data, y_data


def train_classical_model(
    frame: pd.DataFrame,
    model_name: str = "svm",
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainingResult:
    x_data, y_data = split_features_and_labels(frame)
    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=test_size,
        random_state=random_state,
        stratify=y_data,
    )

    if model_name == "svm":
        estimator = SVC(kernel="rbf", class_weight="balanced")
    elif model_name == "knn":
        estimator = KNeighborsClassifier(n_neighbors=5)
    elif model_name == "logreg":
        estimator = LogisticRegression(max_iter=1000, class_weight="balanced")
    else:
        raise ValueError(f"Modelo classico desconhecido: {model_name}")

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    return TrainingResult(
        model_name=model_name,
        accuracy=accuracy_score(y_test, predictions),
        macro_f1=f1_score(y_test, predictions, average="macro"),
        report=classification_report(y_test, predictions),
    )
