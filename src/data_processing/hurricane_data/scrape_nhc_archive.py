"""
- Step 1a: Scrape the NHC public advisory archive
- For each storm in STORMS, fetches archive index page + extracts all advisory links with corresponding UTC timestamps
 
- Output: data/processed/hurricane_data/hurricane_archive_links.csv
"""


import re 
import requests
import pytz
import pandas as pd
from datetime import datetime, timezone
from bs4 import BeautifulSoup, Comment
from pathlib import Path

from config.settings import STORMS, STORM_YEARS
from paths import PATHS


# Constants
BASE_URL = "https://www.nhc.noaa.gov"

COLUMN_NAMES = [
    "forecast_advisories",
    "public_advisories",
    "discussions",
    "wind_speed_probabilities",
]

## col1 and col4 comments are already UTC, but sometimes col2 and col3 are local time
UTC_COLS = {"forecast_advisories", "wind_speed_probabilities"}

TZ_MAP = {
    "UTC": pytz.utc,
    "AST": pytz.timezone("America/Puerto_Rico"),  # UTC-4, no DST
    "EDT": pytz.timezone("America/New_York"),  # UTC-4 (DST)
    "EST": pytz.timezone("America/New_York"),  # UTC-5
    "CDT": pytz.timezone("America/Chicago"),  # UTC-5 (DST)
    "CST": pytz.timezone("America/Chicago"),  # UTC-6
}
 
UTC_COMMENT_RE = re.compile(r"(\d{8})\s+(\d{4})")

OUTPUT_PATH = (
    PATHS["data_processed"]
    / "hurricane_data"
    / "hurricane_archive_links.csv"
)

# helper functions

def make_archive_url(storm_name: str, year: int) -> str:
    return f"{BASE_URL}/archive/{year}/{storm_name.upper()}.shtml?"

def parse_storm_archive(storm_name: str, year: int) -> pd.DataFrame:
    url = make_archive_url(storm_name, year)
    print(f" Fetching: {url}")
    response = requests.get(url)
    response.raise_for_status()
 
    soup = BeautifulSoup(response.text, "lxml")
    table = soup.find("table")
 
    records = []
    current_date = None
    col_index = 0
 
    for td in table.find_all("td"):
        # Date header cell
        if td.get("colspan") == "4":
            current_date = td.get_text(strip=True)
            col_index = 0
            continue
 
        if td.get("valign") != "middle":
            continue
 
        col_name = COLUMN_NAMES[col_index % 4]
        col_index += 1
 
        if not current_date:
            continue

        # map each link to preceding comment timestamp, display timezone
        comment_before_link: dict[int, datetime | None] = {}
        tz_before_link: dict[int, str | None] = {}
        last_comment_dt = None
        last_display_tz = None
 
        for child in td.children:
            if isinstance(child, Comment):
                match = UTC_COMMENT_RE.search(child)
                if match:
                    date_str, time_str = match.group(1), match.group(2)
                    last_comment_dt = datetime(
                        int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]),
                        int(time_str[:2]), int(time_str[2:])
                    )
            elif child.name == "a":
                label = child.get_text(strip=True)
                tz_match = re.search(r"\b(UTC|AST|EDT|EST|CDT|CST)\b", label)
                last_display_tz = tz_match.group(1) if tz_match else None
                comment_before_link[id(child)] = last_comment_dt
                tz_before_link[id(child)] = last_display_tz
 
        for link in td.find_all("a"):
            naive_dt = comment_before_link.get(id(link))
            display_tz = tz_before_link.get(id(link))
            label = link.get_text(strip=True)
 
            utc_dt = None
            if naive_dt is not None:
                if col_name in UTC_COLS:
                    utc_dt = naive_dt.replace(tzinfo=timezone.utc)
                else:
                    if display_tz and display_tz in TZ_MAP:
                        local_tz = TZ_MAP[display_tz]
                        localized = local_tz.localize(naive_dt, is_dst=None)
                        utc_dt = localized.astimezone(pytz.utc)
                    else:
                        # Fallback for update labels with no timezone
                        utc_dt = naive_dt.replace(tzinfo=timezone.utc)
 
            records.append({
                "storm": storm_name,
                "year": year,
                "date": current_date,
                "column": col_name,
                "label": label,
                "utc_datetime": utc_dt,
                "url": BASE_URL + link["href"].rstrip("?"),
            })
 
    return pd.DataFrame(records)

# Main
def run() -> pd.DataFrame:
    all_dfs = []
    for storm_name in STORMS:
        year = STORM_YEARS[storm_name]
        print(f"Scraping {storm_name} ({year})...")
        try:
            df = parse_storm_archive(storm_name, year)
            print(f" -> {len(df)} records found")
            all_dfs.append(df)
        except Exception as e:
            print(f" Failed: {e}")
 
    combined = pd.concat(all_dfs, ignore_index=True)
    combined["utc_datetime"] = pd.to_datetime(combined["utc_datetime"], utc=True)
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(combined)} records to {OUTPUT_PATH}")
    
    return combined
 
 
if __name__ == "__main__":
    run()