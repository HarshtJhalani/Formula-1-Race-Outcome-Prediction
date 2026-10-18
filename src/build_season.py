
import fastf1
import pandas as pd
from pathlib import Path

from src.features import create_features
from src.qualifying import extract_qualifying
from src.race_results import extract_race


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

YEAR = 2023


def build_season(year: int) -> pd.DataFrame:
    """Extract qualifying and race results for a season
    and combine them into one feature dataset."""

    schedule = fastf1.get_event_schedule(year)

    season_features = []
    failed_events = []

    for _, event in schedule.iterrows():

        grand_prix = event["EventName"]

        # Skip pre-season testing
        if "testing" in grand_prix.lower():
            continue

        print(f"\nProcessing {year} {grand_prix}...")

        try:
            # Extract qualifying and race results
            extract_qualifying(year, grand_prix)
            extract_race(year, grand_prix)

            # Build the feature dataset for this event
            event_features = create_features(year, grand_prix)

            season_features.append(event_features)

        except Exception as error:
            failed_events.append((grand_prix, str(error)))
            print(f"Failed: {grand_prix}: {error}")

    # Validate that at least one event succeeded
    if not season_features:
        raise RuntimeError("No events were processed successfully.")

    # Combine all successful events
    combined_features = pd.concat(
        season_features,
        ignore_index=True,
    )

    # Save the season dataset
    output_file = (
        PROCESSED_DATA_DIR / f"features_{year}_season.csv"
    )

    combined_features.to_csv(output_file, index=False)

    print(f"\nSaved season dataset to: {output_file}")
    print(f"Successful events: {len(season_features)}")
    print(f"Failed events: {len(failed_events)}")
    print(f"Total driver-race rows: {len(combined_features)}")

    if failed_events:
        print("\nFailed events:")
        for event_name, error in failed_events:
            print(f"- {event_name}: {error}")

    return combined_features


if __name__ == "__main__":
    season_data = build_season(YEAR)

    print("\nSeason dataset preview:")
    print(season_data.head())
