
import json
import os
import re

import anthropic
import pandas as pd

# Anthropic client 
_client = anthropic.Anthropic()
_MODEL  = "claude-opus-4-5"

INSIGHTS_OUTPUT = "insights.json"


# Helpers 
def _call_claude(system: str, user: str, max_tokens: int = 500) -> dict:
    """Send a prompt to Claude and parse the JSON response."""
    response = _client.messages.create(
        model=_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    raw = response.content[0].text.strip()
    # Strip accidental markdown fences
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


# 1. Cluster profiles

def generate_cluster_profiles(
    df_with_clusters: pd.DataFrame,
    cluster_col: str = "cluster",
    id_col: str = "customer_id",
    target_col: str | None = None,
) -> list[dict]:
    """
    For each cluster, ask Claude to write a plain-English profile:
    a memorable label, a 2-sentence description, a recommended action,
    and a risk level (high / medium / low).
    """
    system = """You are a senior CRM analyst. Given summary statistics for a customer
segment, write a concise profile a non-technical stakeholder can act on.

Respond ONLY with a valid JSON object — no markdown, no preamble.

Schema:
{
  "label":       string,                          // 3-5 word memorable segment name
  "description": string,                          // 2 sentence plain-English profile
  "action":      string,                          // 1 recommended business action
  "risk":        "high" | "medium" | "low"        // churn or miss risk level
}"""

    numeric_cols = [
        c for c in df_with_clusters.select_dtypes(include="number").columns
        if c not in (cluster_col, id_col)
    ]

    profiles = []
    for cid in sorted(df_with_clusters[cluster_col].unique()):
        subset = df_with_clusters[df_with_clusters[cluster_col] == cid]
        stats  = subset[numeric_cols].mean().round(3).to_dict()

        target_note = f"The target variable is '{target_col}'. " if target_col else ""
        user = (
            f"Cluster {cid} — mean feature values:\n"
            f"{json.dumps(stats, indent=2)}\n\n"
            f"{target_note}Write a profile for this customer segment."
        )

        profile = _call_claude(system, user, max_tokens=300)
        profile["cluster_id"] = int(cid)
        profiles.append(profile)
        print(f"  [Insights] Cluster {cid} → '{profile['label']}' ({profile['risk']} risk)")

    return profiles


# 2. Feature importance narrative 

def generate_feature_narrative(
    feature_importance: dict,
    target_col: str,
    top_n: int = 10,
) -> dict:
    """
    Ask Claude to explain the top predictive features in plain English —
    what they are, which direction they push the prediction, and why it matters.
    """
    system = """You are a data scientist explaining model results to a business audience.
Given the top predictive features and their coefficients, explain what drives
the target outcome in plain English.

Respond ONLY with a valid JSON object — no markdown, no preamble.

Schema:
{
  "narrative": string,       // 2-3 sentence plain-English explanation
  "top_drivers": [
    {
      "feature":     string,
      "direction":   "increases" | "decreases",
      "explanation": string    // one sentence, business-friendly
    }
  ]
}"""

    sorted_features = sorted(
        feature_importance.items(), key=lambda x: abs(x[1]), reverse=True
    )[:top_n]

    user = (
        f"Target variable: '{target_col}'\n"
        f"Top feature importances (name → coefficient):\n"
        + "\n".join(f"  {name}: {score:.4f}" for name, score in sorted_features)
    )

    return _call_claude(system, user, max_tokens=400)


# 3. Executive summary 

def generate_executive_summary(
    dataset_stats: dict,
    cluster_profiles: list[dict],
    model_metrics: dict | None = None,
) -> dict:
    """
    Ask Claude to write a short executive summary of the entire analysis —
    a headline finding, top risks, and top opportunities.
    """
    system = """You are a senior analyst presenting findings to a business audience.
Given dataset statistics, customer segment profiles, and model performance,
write a short executive summary.

Respond ONLY with a valid JSON object — no markdown, no preamble.

Schema:
{
  "headline":  string,     // one bold insight sentence (20 words max)
  "summary":   string,     // 3 sentence executive summary
  "risks":     [string],   // up to 3 key business risks
  "opportunities": [string] // up to 3 key business opportunities
}"""

    context = {
        "dataset": dataset_stats,
        "segments": [
            {"label": p["label"], "risk": p["risk"], "action": p["action"]}
            for p in cluster_profiles
        ],
    }
    if model_metrics:
        context["model_performance"] = model_metrics

    user = f"Analysis results:\n{json.dumps(context, indent=2, default=str)}"
    return _call_claude(system, user, max_tokens=400)


# Main entry point 
def generate_all_insights(
    df_with_clusters: pd.DataFrame,
    cluster_col: str = "cluster",
    id_col: str = "customer_id",
    target_col: str | None = None,
    feature_importance: dict | None = None,
    model_metrics: dict | None = None,
) -> dict:
    """
    Run all three insight generators and save the results to insights.json.

    Returns a dict with keys: cluster_profiles, feature_narrative, executive_summary.
    """
    print("\n[Insights] Generating AI insights with Claude...")

    # 1. Cluster profiles
    print("\n[Insights] Profiling customer segments...")
    cluster_profiles = generate_cluster_profiles(
        df_with_clusters, cluster_col, id_col, target_col
    )

    # 2. Feature narrative (only if model was run)
    feature_narrative = None
    if feature_importance and target_col:
        print("\n[Insights] Explaining predictive features...")
        feature_narrative = generate_feature_narrative(feature_importance, target_col)

    # 3. Executive summary
    print("\n[Insights] Writing executive summary...")
    dataset_stats = {
        "total_customers": len(df_with_clusters),
        "n_segments":      int(df_with_clusters[cluster_col].nunique()),
        "target_col":      target_col,
    }
    if target_col and target_col in df_with_clusters.columns:
        dataset_stats["target_rate"] = round(
            float(df_with_clusters[target_col].mean()), 4
        )

    executive_summary = generate_executive_summary(
        dataset_stats, cluster_profiles, model_metrics
    )

    # Bundle and save
    insights = {
        "cluster_profiles":   cluster_profiles,
        "feature_narrative":  feature_narrative,
        "executive_summary":  executive_summary,
    }

    with open(INSIGHTS_OUTPUT, "w") as f:
        json.dump(insights, f, indent=2, default=str)
    print(f"\n[Insights] Saved → {INSIGHTS_OUTPUT}")

    # Print summary to console
    print(f"\n── Executive Summary ────────────────────────────────────────────")
    print(f"  {executive_summary['headline']}")
    print(f"\n  {executive_summary['summary']}")
    print(f"\n── Segment Labels ───────────────────────────────────────────────")
    for p in cluster_profiles:
        print(f"  Cluster {p['cluster_id']}: {p['label']}  [{p['risk']} risk]")
        print(f"    → {p['action']}")

    return insights