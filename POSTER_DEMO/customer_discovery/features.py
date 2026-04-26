from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

@dataclass
class FeatureBundle:
    X: np.ndarray
    feature_names: List[str]
    preprocessor: ColumnTransformer
    categorical_cols: List[str]
    numeric_cols: List[str]

def infer_columns(df: pd.DataFrame, label_col: Optional[str] = None) -> Tuple[List[str], List[str]]:
    cols = df.columns.tolist()
    if label_col and label_col in cols:
        cols = [c for c in cols if c != label_col]

    cat = df[cols].select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num = df[cols].select_dtypes(include=["number"]).columns.tolist()

    for id_like in ["id", "customer_id"]:
        if id_like in num:
            num.remove(id_like)
        if id_like in cat:
            cat.remove(id_like)

    return cat, num

def build_preprocessor(
    categorical_cols: List[str],
    numeric_cols: List[str],
) -> ColumnTransformer:
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    return ColumnTransformer(
        transformers=[
            ("categorical", cat_pipe, categorical_cols),
            ("numeric", num_pipe, numeric_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

def make_features(
    df: pd.DataFrame,
    categorical_cols: Optional[List[str]] = None,
    numeric_cols: Optional[List[str]] = None,
    label_col: Optional[str] = None,
) -> FeatureBundle:
    if categorical_cols is None or numeric_cols is None:
        cat, num = infer_columns(df, label_col=label_col)
        if categorical_cols is None:
            categorical_cols = cat
        if numeric_cols is None:
            numeric_cols = num

    preprocessor = build_preprocessor(categorical_cols, numeric_cols)
    X = preprocessor.fit_transform(df)
    feature_names = preprocessor.get_feature_names_out().tolist()

    return FeatureBundle(
        X=X,
        feature_names=feature_names,
        preprocessor=preprocessor,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
    )
