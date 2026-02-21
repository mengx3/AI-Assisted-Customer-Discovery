import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, dataframe: pd.DataFrame, id_col: str = "customer_id"):
        self.df = dataframe.copy()
        self.id_col = id_col

    def report(self) -> pd.DataFrame:
        df = self.df

        missing_pct = (df.isna().mean() * 100).round(2)
        nunique = df.nunique(dropna=True)

        report_df = pd.DataFrame({
            "dtype": df.dtypes.astype(str),
            "missing_%": missing_pct,
            "unique_count": nunique
        })

        # Add basic numeric stats where applicable
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            report_df.loc[numeric_cols, "min"] = df[numeric_cols].min()
            report_df.loc[numeric_cols, "median"] = df[numeric_cols].median()
            report_df.loc[numeric_cols, "max"] = df[numeric_cols].max()

        return report_df.sort_values(by="missing_%", ascending=False)

    def clean(self) -> pd.DataFrame:
        """
        1) remove duplicate rows
        2) remove whitespace in text columns
        3) convert numeric-looking strings to number
        4) plug missing values (median for numeric, 'Unknown' for categorical)
        """
        df = self.df

        # 1) 
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        if after < before:
            print(f" Remove {before - after} duplicate row(s).")

        # 2) 
        obj_cols = df.select_dtypes(include=["object"]).columns
        for c in obj_cols:
            df[c] = df[c].astype("string").str.strip()

        # 3) 
        for c in obj_cols:
            if c == self.id_col:
                continue
            converted = pd.to_numeric(df[c], errors="coerce")
            # If conversion produces "some" real numbers, treat as numeric
            if converted.notna().mean() > 0.7:  
                df[c] = converted

      
        numeric_cols = df.select_dtypes(include=["number"]).columns
        cat_cols = [c for c in df.columns if c not in numeric_cols]

        # 4) Missing values
        for c in numeric_cols:
            df[c] = df[c].fillna(df[c].median())

        for c in cat_cols:
            df[c] = df[c].fillna("Unknown")

        # 5) 
        if self.id_col in df.columns:
            id_series = df[self.id_col]
            features = df.drop(columns=[self.id_col])
            features = pd.get_dummies(features, drop_first=False)
            df_clean = pd.concat([id_series, features], axis=1)
        else:
            df_clean = pd.get_dummies(df, drop_first=False)

        self.df = df_clean
        return df_clean

