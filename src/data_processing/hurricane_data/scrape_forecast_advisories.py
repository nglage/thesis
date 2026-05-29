"""
- Step 1b: Fetch + parse NHC Tropical Cyclone Forecast/Advisory (TCM) text
    - Reads hurricane_archive_links.csv produced by scrape_nhc_archive.py
    - fetches each forecast advisory page
    - extracts structured variables via regex
 
- Output: data/processed/hurricane_data/hurricane_forecast_advisories.csv
"""

import re
import time
import requests
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup

from paths import PATHS

# Paths
_DATA_DIR   = PATHS["data_processed"] / "hurricane_data"
LINKS_CSV   = _DATA_DIR / "hurricane_archive_links.csv"
OUTPUT_CSV  = _DATA_DIR / "hurricane_forecast_advisories.csv"


# Advisory text parsing
def fetch_advisory_text(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")
    pre = soup.find("pre")
    return pre.get_text() if pre else ""
 
def parse_advisory(text: str, url: str) -> dict:
    """Extract structured variables from advisory plain text using regex."""
    result = {"url": url}

    # advisory number, special advisory flag
    m = re.search(r"(SPECIAL\s+)?FORECAST/ADVISORY NUMBER\s+(\d+)", text)
    if m:
        result["advisory_number"] = int(m.group(2))
        result["is_special"] = bool(m.group(1))

    # storm type
    m = re.search(
        r"(HURRICANE|TROPICAL STORM|TROPICAL DEPRESSION|SUBTROPICAL STORM|SUBTROPICAL DEPRESSION)"
        r"\s+\w+\s+(SPECIAL\s+)?FORECAST/ADVISORY",
        text
    )
    if m:
        result["storm_type"] = m.group(1)

    # timestamp (raw)
    m = re.search(r"(\d{4})\s+UTC\s+\w+\s+(\w+\s+\d+\s+\d{4})", text)
    if m:
        result["advisory_time_utc_raw"] = f"{m.group(1)} UTC {m.group(2)}"

    # current position
    m = re.search(r"CENTER LOCATED NEAR\s+([\d.]+)N\s+([\d.]+)W", text)
    if m:
        result["lat"] = float(m.group(1))
        result["lon"] = -float(m.group(2))

    #position accuracy
    m = re.search(r"POSITION ACCURATE WITHIN\s+(\d+)\s+NM", text)
    if m:
        result["position_accuracy_nm"] = int(m.group(1))

    # current intensity
    m = re.search(r"MAX SUSTAINED WINDS\s+(\d+)\s+KT WITH GUSTS TO\s+(\d+)\s+KT", text)
    if m:
        result["max_wind_kt"] = int(m.group(1))
        result["max_gust_kt"] = int(m.group(2))
    
    # pressure
    m = re.search(r"ESTIMATED MINIMUM CENTRAL PRESSURE\s+(\d+)\s+MB", text)
    if m:
        result["min_pressure_mb"] = int(m.group(1))
    
    # movement
    m = re.search(
        r"PRESENT MOVEMENT TOWARD THE ([\w\-]+) OR (\d+) DEGREES AT\s+(\d+)\s+KT",
        text
    )
    if m:
        result["movement_direction"] = m.group(1)
        result["movement_bearing_deg"] = int(m.group(2))
        result["movement_speed_kt"] = int(m.group(3))
    
    # eye diameter (only for hurricanes)
    m = re.search(r"EYE DIAMETER\s+(\d+)\s+NM", text)
    if m:
        result["eye_diameter_nm"] = int(m.group(1))
    
    # forecasts at multiple time horizons
    forecast_blocks = re.findall(
        r"FORECAST VALID\s+(\d+/\d+Z)\s+([\d.]+)N\s+([\d.]+)W\s*\n"
        r"MAX WIND\s+(\d+)\s+KT",
        text
    )
    forecasts = []
    for timestamp, lat, lon, wind in forecast_blocks:
        forecasts.append({
            "valid_time": timestamp,
            "lat": float(lat),
            "lon": -float(lon),
            "max_wind_kt": int(wind),
        })
    result["forecasts"] = forecasts

    for i, fc in enumerate(forecasts[:3]):
        result[f"forecast_{i+1}_valid_time"] = fc["valid_time"]
        result[f"forecast_{i+1}_lat"]        = fc["lat"]
        result[f"forecast_{i+1}_lon"]        = fc["lon"]
        result[f"forecast_{i+1}_wind_kt"]    = fc["max_wind_kt"]

    # watches and warnings
    m = re.search(
        r"SUMMARY OF WATCHES AND WARNINGS IN EFFECT.*?"
        r"(?=\nA \w+ \w+ MEANS|\nINTERESTS|\nHURRICANE CENTER|\nTROPICAL STORM CENTER)",
        text, re.DOTALL
    )
    watches_text = m.group(0) if m else ""
 
    result["has_hurricane_warning"]      = "HURRICANE WARNING IS IN EFFECT" in watches_text
    result["has_hurricane_watch"]        = "HURRICANE WATCH IS IN EFFECT" in watches_text
    result["has_tropical_storm_warning"] = "TROPICAL STORM WARNING IS IN EFFECT" in watches_text
    result["has_tropical_storm_watch"]   = "TROPICAL STORM WATCH IS IN EFFECT" in watches_text
 
    return result

# Main
def run() -> pd.DataFrame:
    df_links = pd.read_csv(LINKS_CSV)
    df_links["utc_datetime"] = pd.to_datetime(df_links["utc_datetime"], utc=True)
    df_fc = df_links[df_links["column"] == "forecast_advisories"].copy()
 
    print(f"Scraping {len(df_fc)} forecast advisories across "
          f"{df_fc['storm'].nunique()} storms...\n")
 
    records = []
    for i, row in df_fc.iterrows():
        print(f"  [{i+1}/{len(df_fc)}] {row['storm']} advisory — {row['url']}")
        try:
            text = fetch_advisory_text(row["url"])
            parsed = parse_advisory(text, row["url"])
            records.append(parsed)
        except Exception as e:
            print(f" Failed: {e}")
            records.append({"url": row["url"]})
 
        time.sleep(0.3)
 
    df_adv = pd.DataFrame(records)

    # merge storm name, year, utc_datetime from links dataframe
    df_adv = df_adv.merge(
        df_fc[["storm", "year", "utc_datetime", "url"]],
        on="url",
        how="left"
    )
    
    # reorder columns
    id_cols = ["storm", "year", "utc_datetime", "advisory_number", "is_special",
               "storm_type", "advisory_time_utc_raw", "url"]
    other_cols = [c for c in df_adv.columns if c not in id_cols]
    df_adv = df_adv[id_cols + other_cols]
    
    # sort by storm and advisory number
    df_adv = df_adv.sort_values(
        ["storm", "year", "advisory_number"]
    ).reset_index(drop=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_adv.to_csv(OUTPUT_CSV, index=False)
 
    print(f"\nSaved {len(df_adv)} advisories to {OUTPUT_CSV}")
    print(f"\nPer storm:")
    print(df_adv.groupby("storm")["advisory_number"].count().rename("n_advisories"))
 
    return df_adv


if __name__ == "__main__":
    run()