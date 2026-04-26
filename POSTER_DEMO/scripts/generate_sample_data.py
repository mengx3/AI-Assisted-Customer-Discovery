from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def make_sample(n: int = 2000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    regions = ["Northeast", "Midwest", "South", "West"]
    industries = ["SaaS", "E-commerce", "Healthcare", "Finance", "Education", "Manufacturing"]
    company_sizes = ["1-10", "11-50", "51-200", "201-1000", "1000+"]
    channels = ["web", "referral", "ads", "partner", "events"]

    age = rng.integers(18, 70, size=n)
    region = rng.choice(regions, size=n, p=[0.22, 0.24, 0.30, 0.24])
    industry = rng.choice(industries, size=n)
    company_size = rng.choice(company_sizes, size=n, p=[0.20, 0.25, 0.25, 0.20, 0.10])
    acquisition_channel = rng.choice(channels, size=n)

    visits_30d = rng.poisson(lam=8, size=n)
    emails_opened_30d = rng.poisson(lam=3, size=n)
    avg_session_min = np.clip(rng.normal(loc=6.5, scale=2.2, size=n), 0.5, 30)
    support_tickets_90d = rng.poisson(lam=1.2, size=n)
    last_purchase_days = np.clip(rng.exponential(scale=60, size=n), 0, 365).round(0).astype(int)

    orders_12m = np.clip(rng.poisson(lam=2.2, size=n), 0, 30)
    base_spend = np.clip(rng.normal(400, 250, size=n), 0, None)
    spend_boost = (visits_30d * 15) + (emails_opened_30d * 18) + (avg_session_min * 10)
    total_spend_12m = np.clip(base_spend + spend_boost + rng.normal(0, 120, size=n), 0, None).round(2)

    z = (
        -3.0
        + 0.09 * visits_30d
        + 0.15 * emails_opened_30d
        + 0.03 * avg_session_min
        - 0.012 * last_purchase_days
        + 0.0009 * total_spend_12m
        + (acquisition_channel == "referral") * 0.5
        + (industry == "SaaS") * 0.25
        + (company_size == "201-1000") * 0.15
    )
    p = 1 / (1 + np.exp(-z))
    converted = rng.binomial(1, p, size=n)

    df = pd.DataFrame({
        "id": np.arange(1, n + 1),
        "age": age,
        "region": region,
        "industry": industry,
        "company_size": company_size,
        "acquisition_channel": acquisition_channel,
        "visits_30d": visits_30d,
        "emails_opened_30d": emails_opened_30d,
        "avg_session_min": avg_session_min.round(2),
        "support_tickets_90d": support_tickets_90d,
        "last_purchase_days": last_purchase_days,
        "orders_12m": orders_12m,
        "total_spend_12m": total_spend_12m,
        "converted": converted,
    })

    for col in ["age", "industry", "avg_session_min"]:
        mask = rng.random(n) < 0.03
        df.loc[mask, col] = np.nan
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output path (csv or xlsx)")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = make_sample(n=args.n, random_state=args.seed)

    if out.suffix.lower() == ".csv":
        df.to_csv(out, index=False)
    elif out.suffix.lower() in [".xlsx", ".xls"]:
        df.to_excel(out, index=False)
    else:
        raise ValueError("Use .csv or .xlsx output")

    print(f"Wrote sample data: {out} ({len(df)} rows)")

if __name__ == "__main__":
    main()
