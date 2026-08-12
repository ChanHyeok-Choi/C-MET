from __future__ import annotations

import argparse

import pandas as pd
from sklearn.metrics import accuracy_score


def calculate_accemo(df: pd.DataFrame) -> float | None:
    subset = df.dropna(subset=["gt_emotion", "predicted_emotion"])
    if subset.empty:
        return None
    return accuracy_score(subset["gt_emotion"], subset["predicted_emotion"]) * 100


def calculate_mean(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns:
        return None
    values = df[column].dropna()
    if values.empty:
        return None
    return values.astype(float).mean()


def summarize(csv_path: str) -> dict[str, float | None]:
    df = pd.read_csv(csv_path)
    return {
        "FID": calculate_mean(df, "FID"),
        "fvd": calculate_mean(df, "fvd"),
        "Sync_conf": calculate_mean(df, "Sync_conf"),
        "Accemo": calculate_accemo(df),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate the required metric set (FID, FVD, "
        "Sync_conf, Accemo) a CSV accumulates after running the "
        "evaluation pipeline."
    )
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the evaluated CSV.")
    args = parser.parse_args()

    for name, value in summarize(args.csv_path).items():
        if value is None:
            print(f"{name}: (missing — run the corresponding pipeline step first)")
        else:
            print(f"{name}: {value:.4f}")
