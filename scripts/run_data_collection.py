"""
- Run Steps 1a and 1b: scraping NHC advisory archive and parsing advisory text
 
- Output:
  - data/processed/hurricane_data/hurricane_archive_links.csv
  - data/processed/hurricane_data/hurricane_forecast_advisories.csv
"""
 
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
 
from src.data_processing.hurricane_data.scrape_nhc_archive import run as run_archive
from src.data_processing.hurricane_data.scrape_forecast_advisories import run as run_advisories
 
if __name__ == "__main__":
    print("=" * 60)
    print("Step 1a: Scraping NHC archive links")
    print("=" * 60)
    run_archive()
 
    print("\n" + "=" * 60)
    print("Step 1b: Parsing forecast advisories")
    print("=" * 60)
    run_advisories()