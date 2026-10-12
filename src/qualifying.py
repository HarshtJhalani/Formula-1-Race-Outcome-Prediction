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

def extract_qualifying(year, grand_prix):
    
    
    session = fastf1.get_session(year, grand_prix, 'Q')
    session.load(telemetry=False, weather=False, laps=False)
    
    
    results = session.results.copy()
    
    required_columns = ["DriverNumber", "Abbreviation", "FullName", "TeamName", "Position", "Q1", "Q2", "Q3"]
    
    available_col = [column for column in required_columns if column in results.columns]
    
    qualifying_results = results[available_col].copy()
    
    qualifying_results.insert(0, "Year", year)
    qualifying_results.insert(1, 'Grand Prix', grand_prix)
    
    output_file = ( RAW_DATA_DIR / f"qualifying_{year}_{grand_prix.lower().replace(' ', '_')}.csv" ) 
    
    qualifying_results.to_csv(output_file, index=False) 
    print(f"\nQualifying data extracted successfully.") 
    
    return qualifying_results
    
if __name__ == "__main__": 
    data = extract_qualifying( year=2023, grand_prix="Singapore" ) 
    print("\n--- Qualifying Results ---") 
    display_columns = [ "Abbreviation", "FullName", "TeamName", "Position", "Q1", "Q2", "Q3", ] 
    print(data[display_columns].to_string(index=False))
    
    