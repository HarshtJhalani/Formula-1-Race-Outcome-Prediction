from pathlib import Path

import fastf1
import pandas as pd

# Project paths 
PROJECT_ROOT = Path(__file__).resolve().parent.parent 
CACHE_DIR = PROJECT_ROOT / "data" / "cache" 
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

CACHE_DIR.mkdir(parents=True, exist_ok=True) 
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

fastf1.Cache.enable_cache(str(CACHE_DIR))

def extract_race(year, grand_prix):
    
    
    session = fastf1.get_session(year, grand_prix, 'R')
    session.load(telemetry=False, weather=False, laps=False)
    
    
    results = session.results.copy()

    required_columns = ["DriverNumber", "Abbreviation", "FullName", "TeamName", "GridPosition", "Position", "Status", "ClassifiedPosition"] 
    available_col = [column for column in required_columns if column in results.columns]
    
    race_results = results[available_col].copy()
    
    race_results["Podium"] = pd.NA
    valid_position = race_results["Position"].notna()
    
    race_results.loc[valid_position, "Podium"] = (race_results.loc[valid_position, "Position"] <= 3).astype(int)
    
    race_results["Podium"] = race_results["Podium"].astype("Int64")
    
    race_results["PodiumPosition"] = race_results["Position"].where(
    race_results["Position"] <= 3)
    
    race_results.insert(0, "Year", year)
    race_results.insert(1, 'Grand Prix', grand_prix)
    
    output_file = ( RAW_DATA_DIR / f"race_{year}_{grand_prix.lower().replace(' ', '_')}.csv" ) 
    
    race_results.to_csv(output_file, index=False) 
    print(f"\nRace data extracted successfully.") 
    
    return race_results
    
if __name__ == "__main__": 
    data = extract_race( year=2023, grand_prix="Singapore" ) 
    print(f"\n{'-'*50}Race Results{'-'*50}") 
    display_columns = [ "DriverNumber", "Abbreviation", "FullName", "TeamName", "GridPosition", "Position", "Status", "ClassifiedPosition", 'PodiumPosition'] 
    print(data[["Abbreviation", "Position", "Podium"]].to_string(index=False))
    print(data[display_columns].to_string(index=False))
  
    