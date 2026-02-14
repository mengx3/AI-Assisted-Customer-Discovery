import pandas as pd

class DataCleaner:
    def __init__(self, dataframe):
        self.df = dataframe

    def clean(self):
        # Fill numeric missing with median
        numeric_cols = self.df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            self.df[col] = self.df[col].fillna(self.df[col].median())

        # Fill categorical missing with "Unknown"
        categorical_cols = self.df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            self.df[col] = self.df[col].fillna("Unknown")

        # One-hot encode categoricals
        self.df = pd.get_dummies(self.df)

        return self.df

