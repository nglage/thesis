"""
Step 2b: Keyword filter and clean Reddit posts.
 
    - Reads per-storm parquet files from 2a
       - General subreddits (r/hurricane, r/TropicalWeather): keep all.
       - Location subreddits: keep submissions only if storm name or ≥2
           hurricane keywords appear; keep comments only if their parent submission passed the filter
    - Removes deleted posts, bot advisory tables, very short posts, and non-English posts
 
Output: data/processed/reddit/filtered/{Storm}_submissions_filtered.parquet
        data/processed/reddit/filtered/{Storm}_comments_filtered.parquet
        data/processed/reddit/filtered/filtering_summary.csv
"""

import re
from pathlib import Path
 
import pandas as pd
 
from config.settings import STORMS, GENERAL_SUBREDDITS, STORM_NAME_PATTERNS, STRONG_KEYWORDS
from paths import PATHS

# Paths
INPUT_DIR  = PATHS["data_processed"] / "reddit"
OUTPUT_DIR = PATHS["data_processed"] / "reddit" / "filtered"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GENERAL_SUBREDDITS = {"hurricane", "TropicalWeather"}

STORM_REGEXES = {
    name: re.compile(r'\b' + name + r'\b', re.IGNORECASE)
    for name in STORM_NAME_PATTERNS}

BOT_ADVISORY_RE = re.compile(
    r"UTC\s*\|.*?(JST|PDT|AST|CST|EDT|MDT)|"   # forecast tables
    r"ºN\s*\|\s*\d+\.?\d*ºW|"                   # lat/lon tables
    r"Tropical Tidbits\s*\|.*?\|.*?\|",          # tidbits source tables
    re.IGNORECASE | re.DOTALL
    )

REMOVED_RE = re.compile(
    r"^(\[removed\]|\[deleted\]|"
    r"thank you for your submission.*?removed|"
    r"your (post|submission|comment) has been removed)",
    re.IGNORECASE
)

# relevance functions

def is_hurricane_relevant(title: str, selftext: str) -> bool:
    text = f"{title} {selftext}".lower()
    # always keep if storm name appears (whole-word match)
    if any(p.search(text) for p in STORM_REGEXES.values()):
        return True
    # otherwise at least 2 keywords
    return sum(1 for kw in STRONG_KEYWORDS if kw in text) >= 2

def is_non_english(text:str) -> bool:
    if not text or len(text) < 20:
        return False
    words = text.split()
    if not words:
        return False
    non_ascii = sum(1 for w in words if not w.isascii())
    return (non_ascii / len(words)) > 0.30

# cleaning function

def clean_dataframe(df: pd.DataFrame, content_type: str) -> pd.DataFrame:
    n_before = len(df)
    mask = pd.Series(True, index=df.index)
    text_col = "text_clean" if "text_clean" in df.columns else "body"

    # remove removed/deleted/mod messages
    removed_mask = df[text_col].fillna("").apply(
        lambda t: bool(REMOVED_RE.match(str(t).strip()))
    )
    if content_type == "submissions":
        # also check title
        removed_mask |= df.get("title", pd.Series("", index=df.index)).fillna("").apply(
            lambda t: bool(REMOVED_RE.match(str(t).strip()))
        )
    mask &= ~removed_mask

    # remove bot advisory posts (tabular forecast data)
    bot_mask = df[text_col].fillna("").apply(
        lambda t: bool(BOT_ADVISORY_RE.search(str(t)))
    )
    mask &= ~bot_mask

    # remove very short posts
    short_mask = df[text_col].fillna("").apply(
        lambda t: len(str(t).split()) < 15
    )
    mask &= ~short_mask

    # remove non-English posts
    lang_mask = df[text_col].fillna("").apply(
        lambda t: is_non_english(str(t))
    )
    mask &= ~lang_mask

    df_clean = df[mask].copy()

    print(f"    Cleaning ({content_type}): {n_before} -> {len(df_clean)} "
          f"(removed {n_before - len(df_clean)}: "
          f"{removed_mask.sum()} removed/deleted, "
          f"{bot_mask.sum()} bot advisories, "
          f"{short_mask.sum()} too short, "
          f"{lang_mask.sum()} non-English)")

    return df_clean


# Filter functions
def filter_submissions(df: pd.DataFrame) -> pd.DataFrame:
    """
    general subreddits -> keep all
    local subreddits -> keep only hurricane-relevant submissions
    """
    general_mask = df["subreddit"].isin(GENERAL_SUBREDDITS)
    df_general    = df[general_mask].copy()
    df_location   = df[~general_mask].copy()

    if not df_location.empty:
        keyword_mask = df_location.apply(
            lambda row: is_hurricane_relevant(
                row.get("title", ""), row.get("selftext", "")
            ),
            axis=1,
        )
        df_location = df_location[keyword_mask]
    
    return (
        pd.concat([df_general, df_location], ignore_index=True)
        .sort_values("created_utc")
        .reset_index(drop=True)
    )

def filter_comments(df_comments: pd.DataFrame,
                    df_submissions_filtered: pd.DataFrame) -> pd.DataFrame:
    """
    general subreddits -> keep all comments
    location subreddits -> keep only comments whose parent submission passed keyword filter (via link_id)
    """
    general_mask = df_comments["subreddit"].isin(GENERAL_SUBREDDITS)
    df_general = df_comments[general_mask].copy()
    df_location = df_comments[~general_mask].copy()

    if not df_location.empty and not df_submissions_filtered.empty:
        location_subs = df_submissions_filtered[~df_submissions_filtered ["subreddit"].isin(GENERAL_SUBREDDITS)]
        relevant_ids = set(location_subs["id"])
        df_location = df_location[df_location["link_id"]
                                  .str.replace("t3_", "", regex=False)
                                  .isin(relevant_ids)
                                  ].copy()
    return (
        pd.concat([df_general, df_location], ignore_index=True)
        .sort_values("created_utc")
        .reset_index(drop=True)
    )

# storm processing
def process_storm(storm: str) -> dict:
    print(f"\n{'=' * 50}\nProcessing {storm}...")
    stats = {"storm": storm}

    # submissions
    sub_path = INPUT_DIR / f"{storm}_submissions.parquet"
    if not sub_path.exists():
        print(f"  [skip] {sub_path.name} not found")
        return stats
 
    df_sub = pd.read_parquet(sub_path)
    df_sub["created_utc"] = pd.to_datetime(df_sub["created_utc"], utc=True)
    print(f"  Submissions loaded: {len(df_sub)}")
 
    df_sub_filtered = filter_submissions(df_sub)
    df_sub_filtered = clean_dataframe(df_sub_filtered, "submissions")
    print(f"  Submissions after filtering: {len(df_sub_filtered)}")
 
    stats["submissions_before"] = len(df_sub)
    stats["submissions_after"]  = len(df_sub_filtered)

    # comments
    com_path = INPUT_DIR / f"{storm}_comments.parquet"
    if not com_path.exists():
        print(f"  [skip] {com_path.name} not found")
        return stats
 
    df_com = pd.read_parquet(com_path)
    df_com["created_utc"] = pd.to_datetime(df_com["created_utc"], utc=True)
    print(f"  Comments loaded: {len(df_com)}")
    
    df_com_filtered = filter_comments(df_com, df_sub_filtered)
    df_com_filtered = clean_dataframe(df_com_filtered, "comments")
    print(f"  Comments after filtering: {len(df_com_filtered)}")
    
    stats["comments_before"] = len(df_com)
    stats["comments_after"]  = len(df_com_filtered)

    # save for further analysis
    df_sub_filtered.to_parquet(
        OUTPUT_DIR / f"{storm}_submissions_filtered.parquet", index=False
    )
    df_com_filtered.to_parquet(
        OUTPUT_DIR / f"{storm}_comments_filtered.parquet", index=False
    )

    # save for manual inspection
    df_sub_filtered.to_csv(
        OUTPUT_DIR / f"{storm}_submissions_filtered.csv",
        index=False, encoding="utf-8-sig"
    )
    df_com_filtered.to_csv(
        OUTPUT_DIR / f"{storm}_comments_filtered.csv",
        index=False, encoding="utf-8-sig"
    )
 
    print(f"  Saved to {OUTPUT_DIR}")
    return stats


# Main

def run() -> None:
    all_stats = []
    for storm in STORMS:
        stats = process_storm(storm)
        all_stats.append(stats)
 
    print(f"\n{'=' * 50}\n=== Filtering Summary ===")
    df_summary = pd.DataFrame(all_stats)
 
    if "submissions_before" in df_summary.columns:
        df_summary["submissions_kept_%"] = (
            df_summary["submissions_after"] / df_summary["submissions_before"] * 100
        ).round(1)
    if "comments_before" in df_summary.columns:
        df_summary["comments_kept_%"] = (
            df_summary["comments_after"] / df_summary["comments_before"] * 100
        ).round(1)
 
    print(df_summary.to_string(index=False))
    df_summary.to_csv(OUTPUT_DIR / "filtering_summary.csv", index=False)
    print(f"\nDone. All files saved to {OUTPUT_DIR}")
 
if __name__ == "__main__":
    run()
