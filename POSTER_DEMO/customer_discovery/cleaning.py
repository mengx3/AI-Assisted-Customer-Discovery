from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np

@dataclass
class CleaningReport:
    n_rows_in: int
    n_rows_out: int
    columns_in: List[str]
    columns_out: List[str]
    missing_by_col: dict
    duplicates_removed: int

def basic_clean(
    df: pd.DataFrame,
    id_col: str = "id",
    drop_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, CleaningReport]:
    df0 = df.copy()

    df0.columns = [c.strip().lower().replace(" ", "_") for c in df0.columns]

    if drop_columns:
        drop_columns = [c.strip().lower().replace(" ", "_") for c in drop_columns]
        df0 = df0.drop(columns=[c for c in drop_columns if c in df0.columns], errors="ignore")

    df0 = df0.dropna(axis=1, how="all")

    if id_col not in df0.columns:
        df0[id_col] = np.arange(1, len(df0) + 1)

    n_before = len(df0)
    df0 = df0.drop_duplicates(subset=[id_col], keep="first")
    duplicates_removed = n_before - len(df0)

    for c in df0.select_dtypes(include=["object"]).columns:
        df0[c] = df0[c].astype(str).str.strip()

    df0 = df0.replace({"": np.nan, "nan": np.nan, "None": np.nan})

    report = CleaningReport(
        n_rows_in=len(df),
        n_rows_out=len(df0),
        columns_in=list(df.columns),
        columns_out=list(df0.columns),
        missing_by_col=df0.isna().sum().to_dict(),
        duplicates_removed=duplicates_removed,
    )
    return df0, report
