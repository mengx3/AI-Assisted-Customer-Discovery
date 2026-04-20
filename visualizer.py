import os
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Output directory 
CHARTS_DIR = "charts"


def _ensure_dir():
    os.makedirs(CHARTS_DIR, exist_ok=True)


def _save(fig: go.Figure, filename: str):
    _ensure_dir()
    path = os.path.join(CHARTS_DIR, filename)
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"  [Visualizer] Saved → {path}")
    return path


# Shared theme 
_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#1a1a2e",
    plot_bgcolor="#16213e",
    font=dict(family="Inter, sans-serif", color="#e0e0e0"),
    margin=dict(l=60, r=40, t=60, b=60),
)

_CLUSTER_PALETTE = px.colors.qualitative.Vivid


# 1. Cluster size bar chart 

def plot_cluster_sizes(df_with_clusters: pd.DataFrame, cluster_col: str = "cluster") -> str:
    """Bar chart: how many customers are in each cluster."""
    counts = (
        df_with_clusters[cluster_col]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    counts.columns = ["Cluster", "Count"]
    counts["Cluster"] = counts["Cluster"].astype(str)

    fig = px.bar(
        counts,
        x="Cluster",
        y="Count",
        color="Cluster",
        color_discrete_sequence=_CLUSTER_PALETTE,
        title="Customers per Cluster",
        text="Count",
        labels={"Cluster": "Cluster ID", "Count": "Number of Customers"},
    )
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_layout(
        **_LAYOUT,
        showlegend=False,
        xaxis_title="Cluster",
        yaxis_title="Customers",
    )
    return _save(fig, "cluster_sizes.html")


# 2. Scatter plot (top 2 features) 
def plot_cluster_scatter(
    df_with_clusters: pd.DataFrame,
    cluster_col: str = "cluster",
    id_col: str = "customer_id",
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
) -> str:
    """
    Scatter of the two most-variance numeric features, coloured by cluster.
    Auto-picks x_col / y_col if not supplied.
    """
    numeric_cols = [
        c for c in df_with_clusters.select_dtypes(include="number").columns
        if c not in (cluster_col, id_col)
    ]

    if len(numeric_cols) < 2:
        print("  [Visualizer] Not enough numeric columns for scatter — skipping.")
        return ""

    # Auto-pick by variance if not specified
    if x_col is None or x_col not in numeric_cols:
        variances = df_with_clusters[numeric_cols].var().sort_values(ascending=False)
        x_col = variances.index[0]
        y_col = variances.index[1]

    plot_df = df_with_clusters[[x_col, y_col, cluster_col]].copy()
    plot_df[cluster_col] = plot_df[cluster_col].astype(str)

    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color=cluster_col,
        color_discrete_sequence=_CLUSTER_PALETTE,
        title=f"Cluster Distribution: {x_col} vs {y_col}",
        opacity=0.7,
        labels={cluster_col: "Cluster"},
    )
    fig.update_traces(marker=dict(size=6, line=dict(width=0)))
    fig.update_layout(**_LAYOUT)
    return _save(fig, "cluster_scatter.html")


# 3. Cluster profile radar chart 

def plot_cluster_profiles(
    df_with_clusters: pd.DataFrame,
    cluster_col: str = "cluster",
    id_col: str = "customer_id",
    max_features: int = 8,
) -> str:
    """
    Radar (spider) chart comparing normalised mean feature values per cluster.
    Shows up to max_features features selected by highest between-cluster variance.
    """
    numeric_cols = [
        c for c in df_with_clusters.select_dtypes(include="number").columns
        if c not in (cluster_col, id_col)
    ]

    if len(numeric_cols) < 3:
        print("  [Visualizer] Not enough features for radar chart — skipping.")
        return ""

    # Pick features with highest between-cluster variance (most discriminating)
    means = df_with_clusters.groupby(cluster_col)[numeric_cols].mean()
    between_var = means.var(axis=0).sort_values(ascending=False)
    top_features = between_var.index[:max_features].tolist()

    # Normalise means to 0-1 range for radar readability
    profile = df_with_clusters.groupby(cluster_col)[top_features].mean()
    profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)

    fig = go.Figure()
    categories = top_features + [top_features[0]]  # close the polygon

    for i, cluster_id in enumerate(profile_norm.index):
        values = profile_norm.loc[cluster_id].tolist()
        values += [values[0]]  # close
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            name=f"Cluster {cluster_id}",
            line_color=_CLUSTER_PALETTE[i % len(_CLUSTER_PALETTE)],
            opacity=0.75,
        ))

    fig.update_layout(
        **_LAYOUT,
        title="Cluster Feature Profiles (normalised means)",
        polar=dict(
            bgcolor="#16213e",
            radialaxis=dict(visible=True, range=[0, 1], color="#888"),
            angularaxis=dict(color="#aaa"),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    return _save(fig, "cluster_profiles.html")


# 4. Feature importance bar chart 

def plot_feature_importance(
    feature_importance: dict,
    target_col: str = "converted",
    top_n: int = 15,
) -> str:
    """
    Horizontal bar chart of the top_n most important features by absolute coefficient.
    Positive coefficients are coloured blue (increases target), negative red (decreases).
    """
    fi_series = pd.Series(feature_importance).sort_values(key=abs, ascending=False).head(top_n)
    fi_df = fi_series.reset_index()
    fi_df.columns = ["Feature", "Coefficient"]
    fi_df = fi_df.sort_values("Coefficient")  # ascending for horizontal bar

    colors = [
        "#5b9cf6" if v >= 0 else "#f28b82"
        for v in fi_df["Coefficient"]
    ]

    fig = go.Figure(go.Bar(
        x=fi_df["Coefficient"],
        y=fi_df["Feature"],
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
    ))
    fig.update_layout(
        **_LAYOUT,
        title=f"Top {top_n} Predictive Features for '{target_col}'",
        xaxis_title="Coefficient (positive = increases likelihood)",
        yaxis_title="",
        height=max(400, top_n * 32),
    )
    # Zero line
    fig.add_vline(x=0, line_width=1, line_color="#555")
    return _save(fig, "feature_importance.html")


# 5. Elbow curve 
def plot_elbow_curve(
    clean_df: pd.DataFrame,
    cluster_col: str = "cluster",
    id_col: str = "customer_id",
    max_k: int = 8,
    chosen_k: Optional[int] = None,
) -> str:
    """
    Inertia vs k elbow curve so you can visually validate the auto-selected k.
    Marks the chosen k with a vertical dashed line.
    """
    numeric_cols = [
        c for c in clean_df.select_dtypes(include="number").columns
        if c not in (cluster_col, id_col)
    ]
    X = clean_df[numeric_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    max_k = min(max_k, len(X_scaled) - 1)
    ks = list(range(2, max_k + 1))
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ks, y=inertias,
        mode="lines+markers",
        line=dict(color="#5b9cf6", width=2),
        marker=dict(size=8, color="#5b9cf6"),
        name="Inertia",
    ))

    if chosen_k and chosen_k in ks:
        fig.add_vline(
            x=chosen_k,
            line_dash="dash",
            line_color="#f28b82",
            annotation_text=f"  chosen k={chosen_k}",
            annotation_font_color="#f28b82",
        )

    fig.update_layout(
        **_LAYOUT,
        title="Elbow Curve — Inertia vs Number of Clusters",
        xaxis=dict(title="k (number of clusters)", tickmode="linear", dtick=1),
        yaxis_title="Inertia (within-cluster sum of squares)",
    )
    return _save(fig, "elbow_curve.html")



# 6. Feature histograms by cluster 

def plot_feature_histograms(
    df_with_clusters: pd.DataFrame,
    cluster_col: str = "cluster",
    id_col: str = "customer_id",
    max_features: int = 6,
) -> str:
    """
    Histogram grid showing the distribution of the top numeric features,
    split by cluster. Makes it easy to see what makes each segment different.
    Picks the features with the highest between-cluster variance.
    """
    numeric_cols = [
        c for c in df_with_clusters.select_dtypes(include="number").columns
        if c not in (cluster_col, id_col) and df_with_clusters[c].nunique() > 2
    ]

    if len(numeric_cols) < 1:
        print("  [Visualizer] No numeric columns for histograms — skipping.")
        return ""

    # Pick most discriminating features by between-cluster variance
    means = df_with_clusters.groupby(cluster_col)[numeric_cols].mean()
    between_var = means.var(axis=0).sort_values(ascending=False)
    top_features = between_var.index[:max_features].tolist()

    n_cols = 2
    n_rows = -(-len(top_features) // n_cols)  # ceiling division

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=top_features,
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    cluster_ids = sorted(df_with_clusters[cluster_col].unique())

    for i, feature in enumerate(top_features):
        row = i // n_cols + 1
        col = i % n_cols + 1
        for j, cid in enumerate(cluster_ids):
            subset = df_with_clusters[df_with_clusters[cluster_col] == cid][feature].dropna()
            fig.add_trace(
                go.Histogram(
                    x=subset,
                    name=f"Cluster {cid}",
                    marker_color=_CLUSTER_PALETTE[j % len(_CLUSTER_PALETTE)],
                    opacity=0.65,
                    showlegend=(i == 0),  # only show legend once
                    bingroup=i,           # same bin width across clusters per feature
                ),
                row=row,
                col=col,
            )

    fig.update_layout(
        **_LAYOUT,
        title="Feature Distributions by Cluster",
        barmode="overlay",
        height=320 * n_rows,
        legend=dict(orientation="h", yanchor="bottom", y=-0.08),
    )
    return _save(fig, "feature_histograms.html")


# 7. ROC curve 

def plot_roc_curve(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    cv,
    target_col: str = "converted",
) -> str:
    """
    Plot the ROC curve for each CV fold plus the mean ROC curve.
    Each fold gets its own line so you can see how stable the model is.
    A random-chance diagonal reference line is included.
    """
    from sklearn.metrics import roc_curve, auc

    fig = go.Figure()

    fold_aucs = []
    mean_fpr = np.linspace(0, 1, 200)
    tprs = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, probs)
        fold_auc = auc(fpr, tpr)
        fold_aucs.append(fold_auc)

        # Interpolate TPR at common FPR points for mean curve
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"Fold {fold_idx + 1} (AUC={fold_auc:.3f})",
            line=dict(
                color=_CLUSTER_PALETTE[fold_idx % len(_CLUSTER_PALETTE)],
                width=1.5,
            ),
            opacity=0.6,
        ))

    # Mean ROC curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(fold_aucs)
    std_auc  = np.std(fold_aucs)

    fig.add_trace(go.Scatter(
        x=mean_fpr,
        y=mean_tpr,
        mode="lines",
        name=f"Mean ROC (AUC={mean_auc:.3f} ± {std_auc:.3f})",
        line=dict(color="#ffffff", width=3),
    ))

    # Random chance baseline
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random chance",
        line=dict(color="#888888", width=1.5, dash="dash"),
    ))

    fig.update_layout(
        **_LAYOUT,
        title=f"ROC Curve — '{target_col}' prediction",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.02]),
        legend=dict(orientation="v", x=0.62, y=0.08),
    )
    return _save(fig, "roc_curve.html")


# Convenience: generate all charts at once 

def generate_all_charts(
    df_with_clusters: pd.DataFrame,
    clean_df: pd.DataFrame,
    feature_importance: Optional[dict] = None,
    target_col: str = "converted",
    cluster_col: str = "cluster",
    id_col: str = "customer_id",
    chosen_k: Optional[int] = None,
    X: Optional[pd.DataFrame] = None,
    y: Optional[pd.Series] = None,
    model=None,
    cv=None,
) -> list[str]:
    """
    Generate all available charts and return a list of saved file paths.
    Pass X, y, model, cv to also generate the ROC curve.
    """
    print("\n[Visualizer] Generating charts...")
    paths = []

    paths.append(plot_cluster_sizes(df_with_clusters, cluster_col))
    paths.append(plot_cluster_scatter(df_with_clusters, cluster_col, id_col))
    paths.append(plot_cluster_profiles(df_with_clusters, cluster_col, id_col))
    paths.append(plot_elbow_curve(clean_df, cluster_col, id_col, chosen_k=chosen_k))
    paths.append(plot_feature_histograms(df_with_clusters, cluster_col, id_col))

    if feature_importance:
        paths.append(plot_feature_importance(feature_importance, target_col))

    if X is not None and y is not None and model is not None and cv is not None:
        paths.append(plot_roc_curve(X, y, model, cv, target_col))

    paths = [p for p in paths if p]  # remove empty strings from skipped charts
    print(f"\n[Visualizer] {len(paths)} chart(s) saved to ./{CHARTS_DIR}/")
    return paths