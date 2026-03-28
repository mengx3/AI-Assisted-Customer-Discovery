# data_cleaner.py

import pandas as pd
import numpy as np
from typing import Optional


class DataCleaner:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        id_col: str = "customer_id",
        missing_threshold: float = 0.5,
        outlier_cap: bool = True,
        outlier_iqr_factor: float = 1.5,
    ):
       
        self.df = dataframe.copy()
        self.id_col = id_col
        self.missing_threshold = missing_threshold
        self.outlier_cap = outlier_cap
        self.outlier_iqr_factor = outlier_iqr_factor

        # Populated during clean() — exposed for the dashboard / logging
        self.dropped_columns: list[str] = []
        self.capped_columns: list[str] = []
        self.parsed_date_columns: list[str] = []
        self.n_duplicates_removed: int = 0
        self.n_outliers_capped: int = 0

    # Report 

    def report(self) -> pd.DataFrame:
        df = self.df

        missing_pct = (df.isna().mean() * 100).round(2)
        nunique = df.nunique(dropna=True)

        report_df = pd.DataFrame({
            "dtype":         df.dtypes.astype(str),
            "missing_%":     missing_pct,
            "unique_count":  nunique,
        })

        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            report_df.loc[numeric_cols, "min"]    = df[numeric_cols].min()
            report_df.loc[numeric_cols, "median"] = df[numeric_cols].median()
            report_df.loc[numeric_cols, "max"]    = df[numeric_cols].max()

            # Count outliers per column (IQR method)
            outlier_counts = {}
            for c in numeric_cols:
                q1 = df[c].quantile(0.25)
                q3 = df[c].quantile(0.75)
                iqr = q3 - q1
                fence_lo = q1 - self.outlier_iqr_factor * iqr
                fence_hi = q3 + self.outlier_iqr_factor * iqr
                n_out = int(((df[c] < fence_lo) | (df[c] > fence_hi)).sum())
                outlier_counts[c] = n_out
            report_df.loc[numeric_cols, "outliers"] = pd.Series(outlier_counts)

        # Flag columns that will be dropped due to too many missing values
        report_df["will_drop"] = (
            (report_df["missing_%"] / 100 > self.missing_threshold)
            & (report_df.index != self.id_col)
        )

        return report_df.sort_values(by="missing_%", ascending=False)

    # Clean 

    def clean(self) -> pd.DataFrame:
        df = self.df.copy()

        # 1. Remove duplicate rows 
        before = len(df)
        df = df.drop_duplicates()
        self.n_duplicates_removed = before - len(df)
        if self.n_duplicates_removed:
            print(f"  [Cleaner] Removed {self.n_duplicates_removed} duplicate row(s).")

        # 2. Strip whitespace from string columns 
        obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for c in obj_cols:
            df[c] = df[c].astype("string").str.strip()

        # 3. Coerce numeric-looking strings 
        for c in obj_cols:
            if c == self.id_col:
                continue
            converted = pd.to_numeric(df[c], errors="coerce")
            if converted.notna().mean() > 0.7:
                df[c] = converted
                print(f"  [Cleaner] Coerced '{c}' to numeric.")

        # 4. Parse date columns 
        # Re-check object columns after coercion
        remaining_obj = df.select_dtypes(include=["object", "string"]).columns.tolist()
        for c in remaining_obj:
            if c == self.id_col:
                continue
            parsed = _try_parse_dates(df[c])
            if parsed is not None:
                df[c + "_year"]  = parsed.dt.year
                df[c + "_month"] = parsed.dt.month
                df[c + "_day"]   = parsed.dt.day
                df = df.drop(columns=[c])
                self.parsed_date_columns.append(c)
                print(f"  [Cleaner] Parsed '{c}' as date → extracted year/month/day.")

        # 5. Drop high-missing columns 
        missing_pct = df.isna().mean()
        cols_to_drop = [
            c for c in df.columns
            if c != self.id_col and missing_pct[c] > self.missing_threshold
        ]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            self.dropped_columns = cols_to_drop
            print(f"  [Cleaner] Dropped {len(cols_to_drop)} high-missing column(s): {cols_to_drop}")

        # 6. Impute missing values 
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = [c for c in df.columns if c not in numeric_cols]

        for c in numeric_cols:
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].median())

        for c in cat_cols:
            if df[c].isna().any():
                df[c] = df[c].fillna("Unknown")

        # 7. Cap outliers (IQR method) 
        if self.outlier_cap:
            # Don't cap the ID column or binary columns (0/1 only)
            cap_candidates = [
                c for c in numeric_cols
                if c != self.id_col and df[c].nunique() > 2
            ]
            total_capped = 0
            for c in cap_candidates:
                q1 = df[c].quantile(0.25)
                q3 = df[c].quantile(0.75)
                iqr = q3 - q1
                if iqr == 0:
                    continue
                lo = q1 - self.outlier_iqr_factor * iqr
                hi = q3 + self.outlier_iqr_factor * iqr
                n_before = ((df[c] < lo) | (df[c] > hi)).sum()
                if n_before > 0:
                    df[c] = df[c].clip(lower=lo, upper=hi)
                    total_capped += n_before
                    self.capped_columns.append(c)

            self.n_outliers_capped = total_capped
            if total_capped:
                print(
                    f"  [Cleaner] Capped {total_capped} outlier value(s) "
                    f"across {len(self.capped_columns)} column(s)."
                )

        # 8. One-hot encode categorical columns 
        if self.id_col in df.columns:
            id_series = df[self.id_col]
            features  = df.drop(columns=[self.id_col])
            features  = pd.get_dummies(features, drop_first=False)
            df_clean  = pd.concat([id_series, features], axis=1)
        else:
            df_clean = pd.get_dummies(df, drop_first=False)

        self.df = df_clean
        return df_clean

    # Cleaning summary 

    def cleaning_summary(self) -> str:
        """
        Returns a plain-text summary of everything that happened during clean().
        Call this after clean() to see a full log.
        """
        lines = ["[DataCleaner] Cleaning summary:"]
        lines.append(f"  Duplicates removed  : {self.n_duplicates_removed}")
        lines.append(f"  Columns dropped     : {len(self.dropped_columns)}"
                     + (f" {self.dropped_columns}" if self.dropped_columns else ""))
        lines.append(f"  Date cols parsed    : {len(self.parsed_date_columns)}"
                     + (f" {self.parsed_date_columns}" if self.parsed_date_columns else ""))
        lines.append(f"  Outlier values capped: {self.n_outliers_capped}"
                     + (f" (in: {list(dict.fromkeys(self.capped_columns))})" if self.capped_columns else ""))
        return "\n".join(lines)


# Helper 

def _try_parse_dates(series: pd.Series) -> Optional[pd.Series]:
    """
    Try to parse a string Series as dates.
    Returns a datetime Series if >70% of non-null values parse successfully,
    otherwise returns None.
    """
    if series.isna().all():
        return None

    sample = series.dropna().head(50)
    try:
        parsed_sample = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
        success_rate = parsed_sample.notna().mean()
        if success_rate < 0.7:
            return None
        # Parse the full column
        return pd.to_datetime(series, infer_datetime_format=True, errors="coerce")
    except Exception:
        return None