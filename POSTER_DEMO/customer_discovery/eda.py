from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

@dataclass
class EDAPaths:
    summary_path: Path
    missing_path: Path
    hist_dir: Path

def run_basic_eda(df: pd.DataFrame, outdir: str | Path) -> EDAPaths:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary = df.describe(include="all").transpose()
    summary_path = outdir / "summary.csv"
    summary.to_csv(summary_path)

    missing = df.isna().mean().sort_values(ascending=False).to_frame("missing_rate")
    missing_path = outdir / "missing_rate.csv"
    missing.to_csv(missing_path)

    hist_dir = outdir / "hist"
    hist_dir.mkdir(parents=True, exist_ok=True)

    num_cols = df.select_dtypes(include=["number"]).columns
    for c in num_cols:
        fig = plt.figure()
        plt.hist(df[c].dropna().values, bins=30)
        plt.title(f"Histogram: {c}")
        plt.xlabel(c)
        plt.ylabel("count")
        fig.savefig(hist_dir / f"{c}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    return EDAPaths(summary_path=summary_path, missing_path=missing_path, hist_dir=hist_dir)
