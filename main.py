# main.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
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




def main():
    # 1. Load 
    loader = DataLoader(INPUT_FILE)
    df = loader.load_csv()

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

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf",    LogisticRegression(max_iter=2000)),
        ])
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        print("\n── Model Results (Logistic Regression) ──────────────────────")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}")

        # Feature importance (top 10 by absolute coefficient)
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