"""Build pre-race features. Every history feature uses shift(1), so a race never sees its own result."""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

FORM_WINDOW = 5

# Same team, new name across seasons
TEAM_MAP = {
    "Red Bull Racing": "Red Bull",
    "Alpine F1 Team": "Alpine",
    "AlphaTauri": "RB",
    "RB F1 Team": "RB",
    "Alfa Romeo": "Sauber",
}

FEATURES = [
    "GridPosition", "PitLaneStart", "GapToPolePct", "NoQualiTime", "GridVsTeammate",
    "DriverPodiumRate", "DriverAvgFinish", "DriverAvgGrid", "RacesSoFar",
    "TeamPodiumRate", "TeamAvgFinish",
]

# Ranked within each race; value = higher is better?
RANK_COLS = {
    "TeamPodiumRate": True, "DriverPodiumRate": True,
    "TeamAvgFinish": False, "DriverAvgFinish": False, "GapToPolePct": False,
}
MODEL_FEATURES = FEATURES + ["LogGrid", "InvGrid"] + [f"{c}Rank" for c in RANK_COLS]

KEEP = ["Year", "Round", "GrandPrix", "EventDate", "Abbreviation", "Team"] + MODEL_FEATURES + ["FinishingPosition", "Podium"]


def add_quali_features(df):
    df = df.copy()
    pole_time = df.groupby(["Year", "Round"])["BestQualiTime"].transform("min")
    df["GapToPolePct"] = (df["BestQualiTime"] / pole_time - 1) * 100
    # No time set -> slowest in that race; whole race missing (Miami 2025) -> NaN
    race_max = df.groupby(["Year", "Round"])["GapToPolePct"].transform("max")
    df["GapToPolePct"] = df["GapToPolePct"].fillna(race_max)
    df["NoQualiTime"] = df["BestQualiTime"].isna().astype(int)

    teammate_grid = df.groupby(["Year", "Round", "Team"])["GridPosition"].transform("sum") - df["GridPosition"]
    df["GridVsTeammate"] = df["GridPosition"] - teammate_grid
    return df


def past_rolling(series, window=FORM_WINDOW):
    # shift(1): current race excluded
    return series.shift(1).rolling(window, min_periods=1).mean()


def add_driver_form(df):
    df = df.sort_values("EventDate").copy()
    by_driver = df.groupby("Abbreviation")
    df["DriverPodiumRate"] = by_driver["Podium"].transform(past_rolling)
    df["DriverAvgFinish"] = by_driver["FinishingPosition"].transform(past_rolling)
    df["DriverAvgGrid"] = by_driver["GridPosition"].transform(past_rolling)
    df["RacesSoFar"] = by_driver.cumcount()
    return df


def add_team_form(df):
    team_race = (
        df.groupby(["Team", "EventDate"], as_index=False)
        .agg(TeamPodiums=("Podium", "sum"), TeamAvgFinishRace=("FinishingPosition", "mean"))
        .sort_values("EventDate")
    )
    by_team = team_race.groupby("Team")
    team_race["TeamPodiumRate"] = by_team["TeamPodiums"].transform(past_rolling)
    team_race["TeamAvgFinish"] = by_team["TeamAvgFinishRace"].transform(past_rolling)
    return df.merge(team_race[["Team", "EventDate", "TeamPodiumRate", "TeamAvgFinish"]],
                    on=["Team", "EventDate"], how="left", validate="many_to_one")


def add_race_features(df):
    """Non-linear grid + form ranked within the race (pre-race info only)."""
    df = df.copy()
    df["LogGrid"] = np.log(df["GridPosition"])
    df["InvGrid"] = 1 / df["GridPosition"]
    for col, higher_better in RANK_COLS.items():
        df[f"{col}Rank"] = df.groupby(["Year", "Round"])[col].rank(ascending=not higher_better)
    return df


def build_features(raw):
    df = raw.copy()
    df["Team"] = df["TeamName"].replace(TEAM_MAP)
    df = add_quali_features(df)
    df = add_driver_form(df)
    df = add_team_form(df)
    df = add_race_features(df)
    return df.sort_values(["EventDate", "GridPosition"]).reset_index(drop=True)


def check_no_leakage(raw, year=2023, round_number=10):
    """Scramble one race's results; its own features must not change."""
    key = ["Year", "Round", "Abbreviation"]
    race = f"Year == {year} and Round == {round_number}"

    tampered = raw.copy()
    mask = (tampered["Year"] == year) & (tampered["Round"] == round_number)
    tampered.loc[mask, "FinishingPosition"] = tampered.loc[mask, "FinishingPosition"].values[::-1]
    tampered.loc[mask, "Podium"] = (tampered.loc[mask, "FinishingPosition"] <= 3).astype(int)

    before = build_features(raw).query(race).set_index(key)[MODEL_FEATURES].sort_index()
    after = build_features(tampered).query(race).set_index(key)[MODEL_FEATURES].sort_index()
    pd.testing.assert_frame_equal(before, after)


def main():
    raw = pd.read_csv(RAW_DATA_DIR / "races_2022_2025.csv", parse_dates=["EventDate"])
    check_no_leakage(raw)

    features = build_features(raw)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DATA_DIR / "features.csv"
    features[KEEP].to_csv(out, index=False)
    print(f"Leakage check OK. Saved {features[KEEP].shape} to {out}")


if __name__ == "__main__":
    main()
