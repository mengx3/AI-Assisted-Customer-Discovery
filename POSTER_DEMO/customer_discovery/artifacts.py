from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import pandas as pd
from .utils import ensure_dir, save_joblib, save_json

@dataclass
class RunArtifacts:
    outdir: Path
    cleaned_data_path: Path
    segmentation_profile_path: Path
    model_path: Path
    preprocessor_path: Path
    metrics_path: Path
    scores_path: Path

def write_run_artifacts(
    outdir: str | Path,
    cleaned_df: pd.DataFrame,
    segmentation_profile: pd.DataFrame,
    model: Any,
    preprocessor: Any,
    metrics: Dict,
    k_scores: Dict[int, float],
) -> RunArtifacts:
    outdir = ensure_dir(outdir)
    data_dir = ensure_dir(outdir / "data")
    models_dir = ensure_dir(outdir / "models")
    reports_dir = ensure_dir(outdir / "reports")

    cleaned_path = data_dir / "customers_cleaned.csv"
    profile_path = reports_dir / "cluster_profiles.csv"
    model_path = models_dir / "conversion_model.joblib"
    preproc_path = models_dir / "preprocessor.joblib"
    metrics_path = reports_dir / "metrics.json"
    scores_path = reports_dir / "k_silhouette_scores.json"

    cleaned_df.to_csv(cleaned_path, index=False)
    segmentation_profile.to_csv(profile_path, index=False)
    save_joblib(model, model_path)
    save_joblib(preprocessor, preproc_path)
    save_json(metrics, metrics_path)
    save_json({str(k): float(v) for k, v in k_scores.items()}, scores_path)

    return RunArtifacts(
        outdir=outdir,
        cleaned_data_path=cleaned_path,
        segmentation_profile_path=profile_path,
        model_path=model_path,
        preprocessor_path=preproc_path,
        metrics_path=metrics_path,
        scores_path=scores_path,
    )
