from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def create_features(year: int, grand_prix: str) -> pd.DataFrame:


    event_name = grand_prix.lower().replace(" ", "_")

    qualifying_file = (
        RAW_DATA_DIR / f"qualifying_{year}_{event_name}.csv"
    )

    race_file = (
        RAW_DATA_DIR / f"race_{year}_{event_name}.csv"
    )


    if not qualifying_file.exists():
        raise FileNotFoundError(
            f"Qualifying data not found: {qualifying_file}"
        )

    if not race_file.exists():
        raise FileNotFoundError(
            f"Race data not found: {race_file}"
        )

    
    qualifying = pd.read_csv(qualifying_file)
    race = pd.read_csv(race_file)

   
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

    # Merge using event and driver identifiers
    features = qualifying_features.merge(
        race_targets,
        on=["Year", "GrandPrix", "Abbreviation"],
        how="inner",
        validate="one_to_one",
    )

    # Convert qualifying lap times into seconds
    for column in ["Q1Time", "Q2Time", "Q3Time"]:
        if column in features.columns:
            features[column] = pd.to_timedelta(
                features[column],
                errors="coerce",
            ).dt.total_seconds()

    
    features = features.sort_values(
        "QualifyingPosition",
        na_position="last",
    ).reset_index(drop=True)

    
    output_file = (
        PROCESSED_DATA_DIR
        / f"features_{year}_{event_name}.csv"
    )

    features.to_csv(output_file, index=False)

    return features


if __name__ == "__main__":

    feature_data = create_features(
        year=2023,
        grand_prix="Singapore",
    )

    print(f"\n{'-'*50}Processed Feature Dataset{'-'*50}")
    print(feature_data.head())


    if "Podium" in feature_data.columns: 
        
        podium_counts = feature_data["Podium"].value_counts(dropna=False) 
        total_drivers = len(feature_data) 
        
        podium_labels = { 1.0: "Podium Finishes", 0.0: "No Podium Finish", } 
        print("\n" + "-" * 55) 
        print("Race Classification") 
        print("-" * 55) 
        for value, count in podium_counts.items():
             
            if pd.isna(value): 
                label = "Unknown / Missing Finishing Position" 
            else: 
                label = podium_labels.get(value, f"Other: {value}") 
            
            print(f"{label:<42} {count:>3} drivers") 
            
