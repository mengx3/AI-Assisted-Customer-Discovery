# main.py
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from data_loader import DataLoader
from data_cleaner import DataCleaner


# Config
INPUT_FILE = "complex_customer_data.csv"     
CLEANED_OUTPUT = "cleaned_customer_data.csv"
REPORT_OUTPUT = "data_report.csv"

TARGET_COL = "converted"


def main():
    # 1) Load
    loader = DataLoader(INPUT_FILE)
    df = loader.load_csv()

    print("\nOriginal Data (head):")
    print(df.head())

    # 2) Align data first (drop duplicates once)
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if after < before:
        print(f"\nDropped {before - after} duplicate row(s) in raw df (for alignment).")


    # 3) Report (before cleaning)
    cleaner = DataCleaner(df)
    report_df = cleaner.report()

    print("\nData Report (top rows):")
    print(report_df.head(10))

    report_df.to_csv(REPORT_OUTPUT)
    print(f"\nSaved report to: {REPORT_OUTPUT}")

    # 4) Clean + save cleaned data
    clean_df = cleaner.clean()

    print("\nCleaned Data (head):")
    print(clean_df.head())

    clean_df.to_csv(CLEANED_OUTPUT, index=False)
    print(f"\nSaved cleaned data to: {CLEANED_OUTPUT}")

    # simple model 
    if TARGET_COL is not None and TARGET_COL in df.columns:
        # y must come from the same df used for cleaning (after duplicates removed)
        y = df[TARGET_COL].astype(int)

        # X comes from cleaned dataframe
        X = clean_df.copy()

        # Drop ID column if present
        if "customer_id" in X.columns:
            X = X.drop(columns=["customer_id"])

        # IMPORTANT: drop target from X if it exists (prevent leakage)
        if TARGET_COL in X.columns:
            X = X.drop(columns=[TARGET_COL])

        # Safety check
        if len(X) != len(y):
            raise ValueError(f"X and y length mismatch: len(X)={len(X)} vs len(y)={len(y)}")

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Baseline model
        model = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),  # with_mean=False works with one-hot style data
            ("clf", LogisticRegression(max_iter=2000))
        ])

        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        print("\nModel Results (Logistic Regression)")
        print(f"Accuracy: {acc:.4f}")
        print(f"ROC-AUC:   {auc:.4f}")

    else:
        print(
            f"\n Skipping model training because target column '{TARGET_COL}' "
            "was not found in the input CSV.\n"
            "If you want to test end-to-end modeling, add a binary target column (0/1)."
        )


if __name__ == "__main__":
    main()