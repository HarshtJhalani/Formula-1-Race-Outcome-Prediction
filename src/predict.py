"""Predict the podium for one race after qualifying, before the race.

Train on every finished race before it (2022 -> current season), then score
the new race's drivers. Grid = qualifying order unless the official grid is passed in.
"""
import argparse

import fastf1
import pandas as pd

from src.data_collection import CACHE_DIR, RAW_DATA_DIR, best_quali_time, collect_season, drop_non_starters, load_results
from src.features import MODEL_FEATURES, build_features
from src.models import make_logreg

FINAL_C = 0.3  # chosen on 2024 in notebook 04
HISTORY_FILES = ["races_2022_2025.csv", "races_2026_to_date.csv"]
LAST_FULL_SEASON = 2025  # later seasons are fetched up to today


def update_current_season(year):
    """Fetch races already run this season (cached after the first run)."""
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    season = collect_season(year)
    season.to_csv(RAW_DATA_DIR / f"races_{year}_to_date.csv", index=False)
    return season


def load_history():
    files = [RAW_DATA_DIR / f for f in HISTORY_FILES if (RAW_DATA_DIR / f).exists()]
    history = pd.concat([pd.read_csv(f, parse_dates=["EventDate"]) for f in files], ignore_index=True)
    return drop_non_starters(history)


def quali_rows(year, race, grid=None):
    """One row per driver, from qualifying only. Race result columns stay empty.

    race: round number or name ("Mexico", "Monza", "Silverstone"; fuzzy matched).
    """
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    event = fastf1.get_event(year, race)
    round_number = int(event["RoundNumber"])
    quali = load_results(year, round_number, "Q").copy()

    rows = pd.DataFrame({
        "Year": year,
        "Round": round_number,
        "GrandPrix": event["EventName"],
        "EventDate": pd.to_datetime(event["EventDate"]),
        "Abbreviation": quali["Abbreviation"].values,
        "FullName": quali["FullName"].values,
        "TeamName": quali["TeamName"].values,
        "QualifyingPosition": quali["Position"].values,
        "BestQualiTime": quali.apply(best_quali_time, axis=1).values,
    })
    # Official grid (penalties) if known, else qualifying order
    rows["GridPosition"] = rows["Abbreviation"].map(grid) if grid else rows["QualifyingPosition"]
    rows["PitLaneStart"] = 0
    rows["FinishingPosition"] = float("nan")
    rows["Podium"] = float("nan")
    return rows


def predict_race(history, race_rows):
    """Train on history before the race, return per-driver podium probabilities."""
    race_date = race_rows["EventDate"].iloc[0]
    history = history[history["EventDate"] < race_date]

    features = build_features(pd.concat([history, race_rows], ignore_index=True))
    is_target = features["EventDate"] == race_date
    train = features[~is_target]
    target = features[is_target].copy()

    model = make_logreg(FINAL_C).fit(train[MODEL_FEATURES], train["Podium"].astype(int))
    target["PodiumProb"] = model.predict_proba(target[MODEL_FEATURES])[:, 1]
    return (target.sort_values("PodiumProb", ascending=False)
            [["Abbreviation", "Team", "GridPosition", "GapToPolePct", "PodiumProb"]]
            .reset_index(drop=True))


def predict(year, race, grid=None):
    """Podium probabilities for any race from 2022 on. Adds the actual finish if the race is done."""
    if year > LAST_FULL_SEASON:
        update_current_season(year)
    history = load_history()
    rows = quali_rows(year, race, grid)
    if rows.empty or rows["BestQualiTime"].isna().all():
        raise SystemExit(f"No qualifying results yet for {year} {race!r}; run after qualifying.")

    prediction = predict_race(history, rows)
    done = history[(history["Year"] == year) & (history["Round"] == rows["Round"].iloc[0])]
    if not done.empty:
        prediction = prediction.merge(done[["Abbreviation", "FinishingPosition"]], on="Abbreviation", how="left")
    prediction.attrs["race"] = f"{rows['GrandPrix'].iloc[0]} {year}"
    return prediction


def main():
    parser = argparse.ArgumentParser(description="Predict an F1 podium after qualifying.")
    parser.add_argument("year", type=int)
    parser.add_argument("race", help='round number or name, e.g. 19 or "Mexico"')
    args = parser.parse_args()

    race = int(args.race) if args.race.isdigit() else args.race
    prediction = predict(args.year, race)
    print(prediction.attrs["race"])
    print(prediction.round(3).to_string())
    print("\nPredicted podium:", ", ".join(prediction["Abbreviation"].head(3)))
    if "FinishingPosition" in prediction:
        actual = prediction.nsmallest(3, "FinishingPosition")["Abbreviation"]
        print("Actual podium:   ", ", ".join(actual))


if __name__ == "__main__":
    main()
