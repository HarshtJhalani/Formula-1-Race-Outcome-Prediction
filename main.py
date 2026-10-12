import fastf1
from pathlib import Path


CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

fastf1.Cache.enable_cache(str(CACHE_DIR))

# Load the quali session
YEAR = 2023
GRAND_PRIX = "Singapore"
SESSION_TYPE = "Q"

print(f"Loading {YEAR} {GRAND_PRIX} Grand Prix qualifying session...")

session = fastf1.get_session(YEAR, GRAND_PRIX, SESSION_TYPE)
session.load(telemetry=False, weather=False, laps=False)

# Quali results
qualifying_results = session.results

# First five rows
print(f"\n--- Qualifying Results: First Five Drivers ---")
print(qualifying_results.head())

# Columns
print(f"\n{"-"*50}Available Columns{"-"*50}")
print(qualifying_results.columns.tolist())

# Drivers
print(f"\n{"-"*50}Dataset Summary{'-'*50}")
print(f"Total drivers: {len(qualifying_results)}")

# 7. Display selected qualifying information
columns_to_display = [
    column
    for column in ["Abbreviation", "FullName", "TeamName", "Position", "Q1", "Q2", "Q3"]
    if column in qualifying_results.columns
]

print(f"\n{"-"*50}Selected Qualifying Data{"-"*50}")
print(qualifying_results[columns_to_display].to_string(index=False))

# 8. Display the top three qualifiers
print(f"\n{"-"*50}Top Three Qualifiers{"-"*50}")
print(
    qualifying_results
    .sort_values("Position")
    [["Abbreviation", "FullName", "Position"]]
    .head(3)
    .to_string(index=False)
)

