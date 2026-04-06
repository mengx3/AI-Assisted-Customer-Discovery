# main.py
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from data_loader import DataLoader
from data_cleaner import DataCleaner
from visualizer import generate_all_charts


# Config 
INPUT_FILE      = "complex_customer_data.csv"
CLEANED_OUTPUT  = "cleaned_customer_data.csv"
REPORT_OUTPUT   = "data_report.csv"
CLUSTER_OUTPUT  = "clustered_customer_data.csv"

TARGET_COL      = "converted"
ID_COL          = "customer_id"
N_CLUSTERS      = None   # Set to an int to override auto-selection, e.g. N_CLUSTERS = 4
CV_FOLDS        = 5      # Number of cross-validation folds


# Helpers 

def choose_k(X_scaled: np.ndarray, max_k: int = 8) -> int:
    """
    Pick the best number of clusters using the elbow method.
    Finds the k where the inertia drop is largest (biggest second difference).
    Falls back to k=3 if the data is too small.
    """
    max_k = min(max_k, len(X_scaled) - 1)
    if max_k < 2:
        return 2

    ks = range(2, max_k + 1)
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    # Second difference — largest value = sharpest elbow
    diffs = np.diff(inertias)
    second_diffs = np.diff(diffs)

    if len(second_diffs) == 0:
        return 3

    best_k = list(ks)[int(np.argmax(second_diffs)) + 2]
    return int(best_k)


def run_clustering(clean_df: pd.DataFrame, k: int | None = None) -> tuple[pd.DataFrame, int]:
    """
    Run K-Means on the numeric columns of clean_df.

    Returns
    -------
    df_with_clusters : original clean_df with a new 'cluster' column appended
    k                : number of clusters actually used
    """
    # Only use numeric columns for clustering (exclude ID)
    numeric_cols = [
        c for c in clean_df.select_dtypes(include="number").columns
        if c != ID_COL
    ]
    X = clean_df[numeric_cols].values

    # Scale before clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Choose k
    if k is None:
        k = choose_k(X_scaled)
        print(f"\n[Clustering] Auto-selected k = {k} (elbow method)")
    else:
        print(f"\n[Clustering] Using k = {k} (manual override)")

    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X_scaled)

    df_with_clusters = clean_df.copy()
    df_with_clusters["cluster"] = labels

    # Print a quick summary
    counts = pd.Series(labels).value_counts().sort_index()
    print("\n[Clustering] Cluster sizes:")
    for cid, cnt in counts.items():
        print(f"  Cluster {cid}: {cnt:,} customers")

    # Print mean of key numeric columns per cluster
    summary_cols = numeric_cols[:6]   # show first 6 features to keep output tidy
    print("\n[Clustering] Cluster means (first 6 features):")
    print(
        df_with_clusters.groupby("cluster")[summary_cols]
        .mean()
        .round(2)
        .to_string()
    )

    return df_with_clusters, k


# Main 

def main():
    # 1. Load 
    loader = DataLoader(INPUT_FILE)
    df = loader.load()                  # auto-detects format, encoding, delimiter

    print("\n" + loader.summary())
    print("\nOriginal Data (head):")
    print(df.head())

    # 2. Drop duplicates (before cleaning so counts stay aligned) 
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if after < before:
        print(f"\nDropped {before - after} duplicate row(s).")

    # 3. Data quality report (before cleaning) 
    cleaner = DataCleaner(df, id_col=ID_COL)
    report_df = cleaner.report()

    print("\nData Quality Report (top rows):")
    print(report_df.head(10))

    report_df.to_csv(REPORT_OUTPUT)
    print(f"\nSaved report → {REPORT_OUTPUT}")

    # 4. Clean
    clean_df = cleaner.clean()
    print("\n" + cleaner.cleaning_summary())

    print("\nCleaned Data (head):")
    print(clean_df.head())

    clean_df.to_csv(CLEANED_OUTPUT, index=False)
    print(f"Saved cleaned data → {CLEANED_OUTPUT}")

    # 5. Clustering 
    df_with_clusters, k_used = run_clustering(clean_df, k=N_CLUSTERS)

    df_with_clusters.to_csv(CLUSTER_OUTPUT, index=False)
    print(f"\nSaved clustered data → {CLUSTER_OUTPUT}")

    # 6. Predictive model 
    feature_importance = None  # populated below if model runs
    model_metrics = None
    if TARGET_COL is not None and TARGET_COL in df.columns:
        # y from the raw (deduped) df to avoid leakage through one-hot encoding
        y = df[TARGET_COL].astype(int)

        # X from the cleaned df, minus ID and target
        X = clean_df.copy()
        if ID_COL in X.columns:
            X = X.drop(columns=[ID_COL])
        if TARGET_COL in X.columns:
            X = X.drop(columns=[TARGET_COL])

        if len(X) != len(y):
            raise ValueError(
                f"X and y length mismatch: len(X)={len(X)} vs len(y)={len(y)}"
            )

        model = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf",    LogisticRegression(max_iter=2000)),
        ])

        # Cross-validation 
        # StratifiedKFold keeps the same class ratio in every fold
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

        cv_results = cross_validate(
            model, X, y,
            cv=cv,
            scoring=["accuracy", "roc_auc"],
            return_train_score=False,
        )

        acc_scores = cv_results["test_accuracy"]
        auc_scores = cv_results["test_roc_auc"]

        print(f"\n── Model Results (Logistic Regression, {CV_FOLDS}-fold CV) ──────────")
        print(f"  Accuracy : {acc_scores.mean():.4f}  ± {acc_scores.std():.4f}")
        print(f"  ROC-AUC  : {auc_scores.mean():.4f}  ± {auc_scores.std():.4f}")
        print(f"  Per-fold AUC : {[round(s, 4) for s in auc_scores]}")

        model_metrics = {
            "accuracy_mean": round(acc_scores.mean(), 4),
            "accuracy_std":  round(acc_scores.std(), 4),
            "roc_auc_mean":  round(auc_scores.mean(), 4),
            "roc_auc_std":   round(auc_scores.std(), 4),
        }

        # Final model fit on full data (for feature importance) 
        # We refit on all data so we get stable coefficients to interpret.
        # This model is NOT used for the CV metrics above.
        model.fit(X, y)
        coef = model.named_steps["clf"].coef_[0]
        fi_series = pd.Series(coef, index=X.columns)
        fi_abs = fi_series.abs().sort_values(ascending=False)
        print("\n  Top 10 predictive features:")
        print(fi_abs.head(10).round(4).to_string())

        # Store as dict for visualizer
        feature_importance = fi_series.to_dict()

    else:
        print(
            f"\nSkipping model — target column '{TARGET_COL}' not found.\n"
            "Set TARGET_COL to a binary column in your CSV to enable prediction."
        )
        feature_importance = None
        model_metrics = None

    # 7. Generate charts 
    generate_all_charts(
        df_with_clusters=df_with_clusters,
        clean_df=clean_df,
        feature_importance=feature_importance,
        target_col=TARGET_COL,
        cluster_col="cluster",
        id_col=ID_COL,
        chosen_k=k_used,
    )


if __name__ == "__main__":
    main()