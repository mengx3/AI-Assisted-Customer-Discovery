from __future__ import annotations
import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px

from customer_discovery.cleaning import basic_clean
from customer_discovery.features import make_features
from customer_discovery.segmentation import choose_k_by_silhouette, segment_customers, profile_clusters
from customer_discovery.prediction import train_predict_model
from customer_discovery.evaluation import classification_metrics

st.set_page_config(page_title="AI-Assisted Customer Discovery", layout="wide")
st.title("AI-Assisted Customer Discovery")
st.caption("Upload a CSV/XLSX file → clean → segment → predict conversion → visualize insights")

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("Upload customer CSV/XLSX", type=["csv", "xlsx", "xls"])
    st.divider()
    st.header("Segmentation")
    k_min = st.slider("K (min)", 2, 12, 2)
    k_max = st.slider("K (max)", 2, 20, 10)
    st.divider()
    st.header("Prediction")
    label_col = st.text_input("Label column", value="converted")
    model_type = st.selectbox("Model", ["random_forest", "logistic_regression"], index=0)
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)

def show_table(df: pd.DataFrame, title: str):
    st.subheader(title)
    st.dataframe(df, use_container_width=True, height=260)

if uploaded is None:
    st.info("Upload a dataset to begin. Tip: generate sample data using `python scripts/generate_sample_data.py`")
    st.stop()

suffix = Path(uploaded.name).suffix.lower()
df = pd.read_csv(uploaded) if suffix == ".csv" else pd.read_excel(uploaded)

cleaned, report = basic_clean(df, id_col="id")
st.success(f"Loaded {len(df)} rows → cleaned to {len(cleaned)} rows (removed {report.duplicates_removed} duplicates)")

col1, col2 = st.columns(2)
with col1:
    show_table(cleaned.head(20), "Preview (cleaned)")
with col2:
    miss = pd.Series(report.missing_by_col).sort_values(ascending=False).to_frame("missing_count")
    show_table(miss.head(20), "Missing values (top 20 columns)")

st.header("Customer Segmentation")
fb_seg = make_features(cleaned, label_col=label_col)
best_k, scores = choose_k_by_silhouette(fb_seg.X, k_min=k_min, k_max=k_max, random_state=42, n_init=20)
seg_res = segment_customers(fb_seg.X, k=best_k, random_state=42, n_init=20)

cleaned_seg = cleaned.copy()
cleaned_seg["cluster"] = seg_res.labels

score_df = pd.DataFrame({"k": list(scores.keys()), "silhouette": list(scores.values())}).sort_values("k")
st.plotly_chart(px.line(score_df, x="k", y="silhouette", markers=True, title="Silhouette score by K"), use_container_width=True)
st.write(f"Selected **K = {best_k}** (silhouette: {seg_res.silhouette:.3f})")

cluster_counts = cleaned_seg["cluster"].value_counts().sort_index().reset_index()
cluster_counts.columns = ["cluster", "count"]
st.plotly_chart(px.bar(cluster_counts, x="cluster", y="count", title="Customers per cluster"), use_container_width=True)

cluster_profile = profile_clusters(cleaned, seg_res.labels)
show_table(cluster_profile, "Cluster profiles")

st.header("Customer Prioritization (Conversion Prediction)")
if label_col not in cleaned.columns:
    st.warning(f"Label column '{label_col}' not found. Prediction requires a supervised label. Use the sample dataset or add a label.")
    st.stop()

y = cleaned[label_col].fillna(0).astype(int).values
fb_pred = make_features(cleaned, label_col=label_col)
pred_res, _ = train_predict_model(fb_pred.X, y, model_type=model_type, test_size=float(test_size), random_state=42)
metrics = classification_metrics(pred_res.y_true, pred_res.proba, pred_res.y_pred)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{metrics.accuracy:.3f}")
m2.metric("Precision", f"{metrics.precision:.3f}")
m3.metric("Recall", f"{metrics.recall:.3f}")
m4.metric("F1", f"{metrics.f1:.3f}")
st.caption(f"ROC-AUC: {metrics.roc_auc:.3f} | PR-AUC: {metrics.pr_auc:.3f}")

# prioritize all customers using the trained model
model = pred_res.model
proba_all = model.predict_proba(fb_pred.X)[:, 1] if hasattr(model, "predict_proba") else pred_res.proba
ranked = cleaned.copy()
ranked["conversion_score"] = proba_all
ranked = ranked.sort_values("conversion_score", ascending=False)
show_table(ranked.head(50), "Top 50 prioritized customers")
st.plotly_chart(px.histogram(ranked, x="conversion_score", nbins=30, title="Distribution of conversion scores"), use_container_width=True)
