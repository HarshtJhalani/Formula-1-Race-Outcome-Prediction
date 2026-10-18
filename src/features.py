from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def create_features(year: int, grand_prix: str) -> pd.DataFrame:
    """
    Merge qualifying and race results into a driver-level dataset.

    Parameters
    ----------
    year : int
        Formula 1 season.
    grand_prix : str
        Grand Prix name.

    Returns
    -------
    pd.DataFrame
        Combined dataset with pre-race features and race outcome targets.
    """

    event_name = grand_prix.lower().replace(" ", "_")

    # Construct input file paths
    qualifying_file = (
        RAW_DATA_DIR / f"qualifying_{year}_{event_name}.csv"
    )

    race_file = (
        RAW_DATA_DIR / f"race_{year}_{event_name}.csv"
    )

    # Validate input files
    if not qualifying_file.exists():
        raise FileNotFoundError(
            f"Qualifying data not found: {qualifying_file}"
        )

    if not race_file.exists():
        raise FileNotFoundError(
            f"Race data not found: {race_file}"
        )

    # Load datasets
    qualifying = pd.read_csv(qualifying_file)
    race = pd.read_csv(race_file)

    # Standardize column names
    qualifying = qualifying.rename(
        columns={
            "Position": "QualifyingPosition",
            "Grand Prix": "GrandPrix",
            "Q1": "Q1Time",
            "Q2": "Q2Time",
            "Q3": "Q3Time",
        }
    )

    race = race.rename(
        columns={
            "Grand Prix": "GrandPrix",
            "Position": "FinishingPosition",
        }
    )

    # Select qualifying features
    qualifying_columns = [
        "Year",
        "GrandPrix",
        "Abbreviation",
        "FullName",
        "TeamName",
        "QualifyingPosition",
        "Q1Time",
        "Q2Time",
        "Q3Time",
    ]

    qualifying_columns = [
        column
        for column in qualifying_columns
        if column in qualifying.columns
    ]

    qualifying_features = qualifying[qualifying_columns].copy()

    # Select race targets and identifiers
    race_columns = [
        "Year",
        "GrandPrix",
        "Abbreviation",
        "FinishingPosition",
        "Podium",
        "PodiumPosition",
    ]

    race_columns = [
        column
        for column in race_columns
        if column in race.columns
    ]

    race_targets = race[race_columns].copy()

    # Merge on event and driver identifiers
    features = qualifying_features.merge(
        race_targets,
        on=["Year", "GrandPrix", "Abbreviation"],
        how="inner",
        validate="one_to_one",
    )

    # Convert qualifying lap times to seconds
    for column in ["Q1Time", "Q2Time", "Q3Time"]:
        if column in features.columns:
            features[column] = (
                pd.to_timedelta(
                    features[column],
                    errors="coerce",
                ).dt.total_seconds()
            )

    # Sort by qualifying position
    features = features.sort_values(
        "QualifyingPosition",
        na_position="last",
    ).reset_index(drop=True)

    # Save processed dataset
    output_file = (
        PROCESSED_DATA_DIR
        / f"features_{year}_{event_name}.csv"
    )

    features.to_csv(output_file, index=False)

    return features


if __name__ == "__main__":

    # Create the feature dataset
    feature_data = create_features(
        year=2023,
        grand_prix="Singapore",
    )

    # Display processed dataset
    print("\n" + "-" * 22 + " Processed Feature Dataset " + "-" * 22)
    print(feature_data.head().to_string(index=False))

    print("\n" + "-" * 22 + " Dataset Summary " + "-" * 22)
    print(f"Season: {feature_data['Year'].iloc[0]}")
    print(f"Grand Prix: {feature_data['GrandPrix'].iloc[0]}")
    print(f"Total drivers: {len(feature_data)}")
    print(f"Total columns: {len(feature_data.columns)}")

    print("\nAvailable columns:")
    print(feature_data.columns.tolist())

    # Display race classification
    if "Podium" in feature_data.columns:

        podium_counts = feature_data["Podium"].value_counts(
            dropna=False
        )

        total_drivers = len(feature_data)

        podium_labels = {
            1.0: "Podium Finish (P1, P2, or P3)",
            0.0: "No Podium Finish",
        }

        print("\n" + "-" * 60)
        print("PODIUM OUTCOME DISTRIBUTION")
        print("-" * 60)

        for value, count in podium_counts.items():

            if pd.isna(value):
                label = "Unknown / Missing Finishing Position"
            else:
                label = podium_labels.get(
                    value,
                    f"Other: {value}",
                )

            percentage = count / total_drivers * 100

            print(
                f"{label:<42} "
                f"{count:>3} drivers "
                f"({percentage:5.1f}%)"
            )

        print("-" * 60)
        print(f"{'Total drivers':<42} {total_drivers:>3}")
        print("-" * 60)


