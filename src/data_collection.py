"""Collect qualifying + race results for 2022-2025 into one table (one row per driver per race)."""
from pathlib import Path

import fastf1
import pandas as pd
from fastf1.req import RateLimitExceededError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
SEASON_DIR = RAW_DATA_DIR / "seasons"
SEASONS = [2022, 2023, 2024, 2025]

# Drivers who withdrew after quali but FastF1 still gives a grid slot
NON_STARTERS = [(2023, "Singapore Grand Prix", "STR")]


def load_results(year, round_number, session_type):
    session = fastf1.get_session(year, round_number, session_type)
    session.load(laps=False, telemetry=False, weather=False, messages=False)
    return session.results


def best_quali_time(row):
    times = pd.to_timedelta(row[["Q1", "Q2", "Q3"]], errors="coerce")
    return times.min().total_seconds() if times.notna().any() else float("nan")


def collect_race(year, round_number, event_name, event_date):
    quali = load_results(year, round_number, "Q")
    race = load_results(year, round_number, "R")

    quali = quali[["Abbreviation", "Position", "Q1", "Q2", "Q3"]].copy()
    quali["BestQualiTime"] = quali.apply(best_quali_time, axis=1)
    quali = quali.rename(columns={"Position": "QualifyingPosition"})
    quali = quali[["Abbreviation", "QualifyingPosition", "BestQualiTime"]]

    race = race[["Abbreviation", "FullName", "TeamName", "GridPosition", "Position", "Status"]]
    race = race.rename(columns={"Position": "FinishingPosition"})

    data = race.merge(quali, on="Abbreviation", how="left", validate="one_to_one")

    # Grid 0 = pit lane start; put behind last grid slot
    pit_lane = data["GridPosition"] == 0
    data.loc[pit_lane, "GridPosition"] = data["GridPosition"].max() + 1
    data["PitLaneStart"] = pit_lane.astype(int)

    data["Podium"] = (data["FinishingPosition"] <= 3).astype(int)

    data.insert(0, "Year", year)
    data.insert(1, "Round", round_number)
    data.insert(2, "GrandPrix", event_name)
    data.insert(3, "EventDate", pd.to_datetime(event_date))
    return data


def collect_season(year):
    path = SEASON_DIR / f"races_{year}.csv"
    if path.exists():
        print(f"{year}: loaded from disk")
        return pd.read_csv(path, parse_dates=["EventDate"])

    schedule = fastf1.get_event_schedule(year, include_testing=False)
    # Current season: only races already run (cache makes re-runs cheap)
    completed = schedule[schedule["EventDate"] < pd.Timestamp.now().normalize()]
    rows, failed = [], []
    for _, event in completed.iterrows():
        try:
            rows.append(collect_race(year, event["RoundNumber"], event["EventName"], event["EventDate"]))
        except RateLimitExceededError:
            raise  # stop the run; re-run in an hour, cache keeps progress
        except Exception as error:
            failed.append((event["EventName"], str(error)))
    print(f"{year}: {len(rows)} ok, {len(failed)} failed {failed}")

    season = pd.concat(rows, ignore_index=True)
    if not failed and len(completed) == len(schedule):
        season.to_csv(path, index=False)  # only save full, finished seasons
    return season


def drop_non_starters(data):
    data = data[data["GridPosition"].notna()]
    for year, grand_prix, driver in NON_STARTERS:
        mask = (data["Year"] == year) & (data["GrandPrix"] == grand_prix) & (data["Abbreviation"] == driver)
        data = data[~mask]
    return data.reset_index(drop=True)


def check(data):
    podiums_per_race = data.groupby(["Year", "Round"])["Podium"].sum()
    bad = podiums_per_race[podiums_per_race != 3]
    assert bad.empty, f"Races without exactly 3 podiums:\n{bad}"
    assert (data["GridPosition"] > 0).all(), "Grid 0 rows left"


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    SEASON_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    data = pd.concat([collect_season(y) for y in SEASONS], ignore_index=True)
    data = data.sort_values(["EventDate", "GridPosition"]).reset_index(drop=True)
    data = drop_non_starters(data)
    check(data)

    out = RAW_DATA_DIR / "races_2022_2025.csv"
    data.to_csv(out, index=False)
    print(f"Saved {data.shape} to {out}")


if __name__ == "__main__":
    main()
