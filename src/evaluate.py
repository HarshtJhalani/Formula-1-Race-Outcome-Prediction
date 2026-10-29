"""Time split, grid top-3 baseline and metrics (podium hits per race, Brier score)."""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_YEARS = [2022, 2023]
VALID_YEARS = [2024]
TEST_YEARS = [2025]  # locked until the final notebook


def load_features():
    return pd.read_csv(PROCESSED_DATA_DIR / "features.csv", parse_dates=["EventDate"])


def time_split(features, include_test=False):
    train = features[features["Year"].isin(TRAIN_YEARS)].copy()
    valid = features[features["Year"].isin(VALID_YEARS)].copy()
    assert train["EventDate"].max() < valid["EventDate"].min()
    if not include_test:
        return train, valid
    test = features[features["Year"].isin(TEST_YEARS)].copy()
    assert valid["EventDate"].max() < test["EventDate"].min()
    return train, valid, test


def podium_hits(df, prob_col):
    """Hits per race: top 3 by probability vs actual podium (0-3)."""
    top3 = df.sort_values(prob_col, ascending=False).groupby(["Year", "Round"]).head(3)
    return top3.groupby(["Year", "Round"])["Podium"].sum()


def brier(df, prob_col):
    return np.mean((df[prob_col] - df["Podium"]) ** 2)


def evaluate(df, prob_col):
    hits = podium_hits(df, prob_col)
    return {
        "hits_per_race": round(hits.mean(), 3),
        "perfect_races_pct": round((hits == 3).mean() * 100, 1),
        "brier": round(brier(df, prob_col), 4),
    }


def fit_grid_baseline(train):
    """Podium rate per grid slot, learned from train only."""
    return train.groupby("GridPosition")["Podium"].mean()


def baseline_prob(df, grid_rate):
    # Grid slots never seen in train -> 0
    return df["GridPosition"].map(grid_rate).fillna(0)


def main():
    train, valid = time_split(load_features())
    grid_rate = fit_grid_baseline(train)
    train["BaselineProb"] = baseline_prob(train, grid_rate)
    valid["BaselineProb"] = baseline_prob(valid, grid_rate)

    print(pd.DataFrame({"train 2022-23": evaluate(train, "BaselineProb"),
                        "valid 2024": evaluate(valid, "BaselineProb")}))


if __name__ == "__main__":
    main()
