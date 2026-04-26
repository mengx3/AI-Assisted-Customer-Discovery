from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

@dataclass
class SegmentationResult:
    k: int
    model: KMeans
    labels: np.ndarray
    silhouette: float

def choose_k_by_silhouette(X: np.ndarray, k_min: int = 2, k_max: int = 10, random_state: int = 42, n_init: int = 20) -> Tuple[int, Dict[int, float]]:
    scores: Dict[int, float] = {}
    best_k = k_min
    best_score = -1.0
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = km.fit_predict(X)
        try:
            s = silhouette_score(X, labels)
        except Exception:
            s = float("nan")
        scores[k] = float(s)
        if np.isfinite(s) and s > best_score:
            best_score = float(s)
            best_k = k
    return best_k, scores

def segment_customers(X: np.ndarray, k: int, random_state: int = 42, n_init: int = 20) -> SegmentationResult:
    model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    labels = model.fit_predict(X)
    try:
        sil = float(silhouette_score(X, labels))
    except Exception:
        sil = float("nan")
    return SegmentationResult(k=k, model=model, labels=labels, silhouette=sil)

def profile_clusters(
    df_original: pd.DataFrame,
    cluster_labels: np.ndarray,
    numeric_cols: Optional[List[str]] = None,
    top_categories: int = 3,
) -> pd.DataFrame:
    df = df_original.copy()
    df["cluster"] = cluster_labels

    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ["cluster"]]

    num_profile = df.groupby("cluster")[numeric_cols].agg(["mean", "median"]).round(3)
    num_profile.columns = ["__".join(col).strip() for col in num_profile.columns.values]

    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ["cluster"]]

    cat_summary_rows = []
    for cl in sorted(df["cluster"].unique()):
        dcl = df[df["cluster"] == cl]
        row = {"cluster": int(cl), "n_customers": int(len(dcl))}
        for c in cat_cols:
            vc = dcl[c].value_counts(dropna=True).head(top_categories)
            row[f"{c}__top"] = "; ".join([f"{idx} ({int(v)})" for idx, v in vc.items()])
        cat_summary_rows.append(row)
    cat_profile = pd.DataFrame(cat_summary_rows).set_index("cluster")

    out = cat_profile.join(num_profile, how="left")
    return out.reset_index()
