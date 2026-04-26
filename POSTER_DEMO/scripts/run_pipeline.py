from __future__ import annotations
import argparse
from pathlib import Path

from customer_discovery.config import PipelineConfig
from customer_discovery.io import read_table
from customer_discovery.cleaning import basic_clean
from customer_discovery.eda import run_basic_eda
from customer_discovery.features import make_features
from customer_discovery.segmentation import choose_k_by_silhouette, segment_customers, profile_clusters
from customer_discovery.prediction import train_predict_model
from customer_discovery.evaluation import classification_metrics
from customer_discovery.artifacts import write_run_artifacts
from customer_discovery.utils import save_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input customer data file (.csv or .xlsx)")
    ap.add_argument("--outdir", default="outputs", help="Output directory")
    ap.add_argument("--label-col", default="converted", help="Label column for prediction")
    ap.add_argument("--model-type", default="random_forest", choices=["random_forest", "logistic_regression"])
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = PipelineConfig(
        random_state=args.seed,
        k_min=args.k_min,
        k_max=args.k_max,
        label_col=args.label_col,
        model_type=args.model_type,
    )

    df = read_table(args.input)
    cleaned, report = basic_clean(df, id_col="id")

    outdir = Path(args.outdir)
    (outdir / "reports").mkdir(parents=True, exist_ok=True)

    save_json(report.__dict__, outdir / "reports" / "cleaning_report.json")
    run_basic_eda(cleaned, outdir / "reports" / "eda")

    fb_seg = make_features(cleaned, label_col=cfg.label_col)
    best_k, scores = choose_k_by_silhouette(
        fb_seg.X, k_min=cfg.k_min, k_max=cfg.k_max, random_state=cfg.random_state, n_init=cfg.kmeans_n_init
    )
    seg_res = segment_customers(fb_seg.X, k=best_k, random_state=cfg.random_state, n_init=cfg.kmeans_n_init)

    cleaned_with_cluster = cleaned.copy()
    cleaned_with_cluster["cluster"] = seg_res.labels

    cluster_profile = profile_clusters(cleaned, seg_res.labels)

    if cfg.label_col not in cleaned.columns:
        raise ValueError(f"Label column '{cfg.label_col}' not found. Provide it, or generate sample data first.")
    y = cleaned[cfg.label_col].fillna(0).astype(int).values

    fb_pred = make_features(cleaned, label_col=cfg.label_col)
    pred_res, _ = train_predict_model(
        fb_pred.X, y, model_type=cfg.model_type, test_size=cfg.test_size, random_state=cfg.random_state
    )
    metrics = classification_metrics(pred_res.y_true, pred_res.proba, pred_res.y_pred).__dict__

    artifacts = write_run_artifacts(
        outdir=outdir,
        cleaned_df=cleaned_with_cluster,
        segmentation_profile=cluster_profile,
        model=pred_res.model,
        preprocessor=fb_pred.preprocessor,
        metrics=metrics,
        k_scores=scores,
    )

    print("✅ Pipeline complete.")
    print(f"- Cleaned data: {artifacts.cleaned_data_path}")
    print(f"- Cluster profiles: {artifacts.segmentation_profile_path}")
    print(f"- Model: {artifacts.model_path}")
    print(f"- Metrics: {artifacts.metrics_path}")
    print(f"- K scores: {artifacts.scores_path}")

if __name__ == "__main__":
    main()
