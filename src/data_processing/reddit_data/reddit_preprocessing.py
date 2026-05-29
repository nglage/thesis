"""
- Step 2a: Time-filter Reddit data by storm advisory windows.
 
    - Reads raw Reddit NDJSON files from the Academic Torrents ZIP archive
    - filters posts/comments to each storm's advisory window (± buffer days)
    - saves per-storm parquet files
 
- Output: 
    - data/processed/reddit/{Storm}_comments.parquet
    - data/processed/reddit/{Storm}_submissions.parquet
    - data/processed/reddit/preprocessing_summary.csv
"""

import re
import json
import zipfile
import pandas as pd
from pathlib import Path

from config.settings import SUBREDDIT_STORM_MAP, PRE_BUFFER_DAYS, POST_BUFFER_DAYS
from paths import PATHS

# Paths
ZIP_PATH = PATHS["reddit_zip"]
ADVISORY_PATH = PATHS["data_processed"] / "hurricane_data" / "hurricane_archive_links.csv"
OUTPUT_DIR = PATHS["data_processed"] / "reddit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Derive storm windows from NHC advisory data

def get_storm_windows(advisory_path: Path) -> dict:
    df = pd.read_csv(advisory_path)
    df["utc_datetime"] = pd.to_datetime(df["utc_datetime"], utc=True)

    windows = {}
    for storm, group in df.groupby("storm"):
        first = group["utc_datetime"].min()
        last  = group["utc_datetime"].max()
        windows[storm] = (
            first - pd.Timedelta(days=PRE_BUFFER_DAYS),
            last  + pd.Timedelta(days=POST_BUFFER_DAYS),
        )
        print(f"  {storm}: {windows[storm][0].date()} → {windows[storm][1].date()}")
 
    return windows


# Text cleaning
DELETED_MARKERS = {"[deleted]", "[removed]", "", None}
WHITESPACE_RE = re.compile(r"\s+")
LINK_RE = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
URL_RE = re.compile(r"http\S+")
SUBREDDIT_RE = re.compile(r"/?r/\w+")
USER_RE = re.compile(r"/?u/\w+")
 
def clean_text(text: str) -> str:
    if not text or text.strip() in DELETED_MARKERS:
        return ""
    text = LINK_RE.sub(r"\1", text)
    text = URL_RE.sub("", text)
    text = SUBREDDIT_RE.sub("", text)
    text = USER_RE.sub("", text)
    return WHITESPACE_RE.sub(" ", text).strip()
 
def is_valid_author(author: str) -> bool:
    return author not in {"[deleted]", "[removed]", "AutoModerator", None, ""}

 
def process_comments(records: list[dict], subreddit: str) -> pd.DataFrame:
    rows = []
    for r in records:
        body = r.get("body", "")
        if not body or body.strip() in DELETED_MARKERS:
            continue
        if not is_valid_author(r.get("author", "")):
            continue
        cleaned = clean_text(body)
        if not cleaned:
            continue
        rows.append({
            "id":               r.get("id"),
            "author":           r.get("author"),
            "created_utc":      pd.Timestamp(int(r["created_utc"]), unit="s", tz="UTC"),
            "subreddit":        subreddit,
            "body":             body,           # raw text
            "body_clean":       cleaned,        # cleaned text for NLP
            "score":            r.get("score", 0),
            "controversiality": r.get("controversiality", 0),
            "parent_id":        r.get("parent_id"),
            "link_id":          r.get("link_id"),
            "type":             "comment",
        })
    return pd.DataFrame(rows)


def process_submissions(records: list[dict], subreddit: str) -> pd.DataFrame:
    rows = []
    for r in records:
        title    = r.get("title", "")
        selftext = r.get("selftext", "")
        if not title or title.strip() in DELETED_MARKERS:
            continue
        if not is_valid_author(r.get("author", "")):
            continue
        # Combine title + body for NLP; keep both raw fields separately
        combined_clean = clean_text(f"{title} {selftext}".strip())
        rows.append({
            "id":           r.get("id"),
            "author":       r.get("author"),
            "created_utc":  pd.Timestamp(int(r["created_utc"]), unit="s", tz="UTC"),
            "subreddit":    subreddit,
            "title":        title,
            "selftext":     selftext,
            "text_clean":   combined_clean,    # cleaned title + body for NLP
            "score":        r.get("score", 0),
            "num_comments": r.get("num_comments", 0),
            "is_self":      r.get("is_self", False),
            "domain":       r.get("domain", ""),
            "url":          r.get("url", ""),
            "permalink":    r.get("permalink", ""),
            "type":         "submission",
        })
    return pd.DataFrame(rows)


def _read_ndjson_from_zip(z: zipfile.ZipFile, filename: str) -> list[dict]:
    records = []
    with z.open(filename) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records
 

def filter_to_window(df: pd.DataFrame, start: pd.Timestamp,
                     end: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df
    return df[(df["created_utc"] >= start) & (df["created_utc"] <= end)].copy()

# Main
def run() -> None:
    print("Deriving storm windows from NHC advisory data...")
    storm_windows = get_storm_windows(ADVISORY_PATH)
 
    storm_comments    = {s: [] for s in storm_windows}
    storm_submissions = {s: [] for s in storm_windows}
 
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        available = set(z.namelist())
 
        for subreddit, storms in SUBREDDIT_STORM_MAP.items():

            for file_type, processor, bucket in [
                ("comments",    process_comments,    storm_comments),
                ("submissions", process_submissions, storm_submissions),
            ]:
                filename = f"{subreddit}_{file_type}"
                if filename not in available:
                    print(f"  [skip] {filename} not found in zip")
                    continue
 
                print(f"\nReading {filename}...")
                records = _read_ndjson_from_zip(z, filename)
                df = processor(records, subreddit)
                print(f"  {len(df)} valid {file_type} total")
 
                for storm in storms:
                    if storm not in storm_windows:
                        continue
                    start, end = storm_windows[storm]
                    filtered = filter_to_window(df, start, end)
                    if not filtered.empty:
                        filtered["storm"] = storm
                        bucket[storm].append(filtered)
                        print(f" -> {storm}: {len(filtered)} {file_type} in window")
                    else:
                        print(f" -> {storm}: 0 {file_type} in window")

    print("\nSaving outputs...")
    summary = []
 
    for storm in storm_windows:
        for file_type, bucket in [("comments", storm_comments), ("submissions", storm_submissions)]:
            if not bucket[storm]:
                print(f"  [warning] No {file_type} found for {storm}")
                continue
 
            df_out = (
                pd.concat(bucket[storm], ignore_index=True)
                .drop_duplicates("id")
                .sort_values("created_utc")
                .reset_index(drop=True)
            )
            out = OUTPUT_DIR / f"{storm}_{file_type}.parquet"
            df_out.to_parquet(out, index=False)
            summary.append({
                "storm":        storm,
                "type":         file_type,
                "n":            len(df_out),
                "subreddits":   df_out["subreddit"].nunique(),
                "window_start": storm_windows[storm][0].date(),
                "window_end":   storm_windows[storm][1].date(),
                "file":         out.name,
            })
            print(f"  Saved {out.name}: {len(df_out)} rows from {df_out['subreddit'].nunique()} subreddits")
   
    df_summary = pd.DataFrame(summary)
    print("\n=== Summary ===")
    print(df_summary.to_string(index=False))
    df_summary.to_csv(OUTPUT_DIR / "preprocessing_summary.csv", index=False)
    print(f"\nDone. Files saved to {OUTPUT_DIR}")
 

if __name__ == "__main__":
    run()