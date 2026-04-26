from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

@dataclass
class PredictionResult:
    model_name: str
    model: object
    proba: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray

def train_predict_model(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "random_forest",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[PredictionResult, Dict[str, object]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
    )

    if model_type == "logistic_regression":
        model = LogisticRegression(max_iter=2000)
        model_name = "LogisticRegression"
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
        model_name = "RandomForest"
    else:
        raise ValueError("model_type must be 'logistic_regression' or 'random_forest'")

    model.fit(X_train, y_train)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
    else:
        scores = model.decision_function(X_test)
        proba = 1 / (1 + np.exp(-scores))

    y_pred = (proba >= 0.5).astype(int)
    return PredictionResult(model_name=model_name, model=model, proba=proba, y_true=y_test, y_pred=y_pred), {"X_test": X_test, "y_test": y_test}
