"""
- Step 4: Merge scored Reddit posts with NHC advisory data
 
- Each post is matched to the most recent advisory issued at or before its timestamp (backward merge_asof)
- Advisory-derived delta variables, warning flags, and storm phase assignment are added.
 
- Output: data/processed/analysis/{Storm}_merged.parquet
   - data/processed/analysis/all_storms_merged.parquet
   - data/processed/analysis/merge_summary.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
 
from config.settings import STORMS
from paths import PATHS

# Paths
SCORED_DIR    = PATHS["data_processed"] / "reddit" / "scored"
ADVISORY_PATH = PATHS["data_processed"] / "hurricane_data" / "hurricane_forecast_advisories.csv"
OUTPUT_DIR    = PATHS["data_processed"] / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# advisory variable helpers
def wind_to_category(wind_kt):
    if pd.isna(wind_kt): return np.nan
    w = float(wind_kt)
    if w >= 137: return 5
    if w >= 113: return 4
    if w >= 96:  return 3
    if w >= 83:  return 2
    if w >= 64:  return 1
    return 0


def prepare_advisories(path):
    df = pd.read_csv(path)
    df["utc_datetime"] = pd.to_datetime(df["utc_datetime"], utc=True)
    df = df.sort_values(["storm", "utc_datetime"]).reset_index(drop=True)
    df["storm_category"] = df["max_wind_kt"].apply(wind_to_category)
    df["delta_wind_kt"] = df.groupby("storm")["max_wind_kt"].diff()
    df["delta_pressure_mb"] = df.groupby("storm")["min_pressure_mb"].diff()
    df["delta_category"] = df.groupby("storm")["storm_category"].diff()
    df["delta_forecast_1_wind"] = df.groupby("storm")["forecast_1_wind_kt"].diff()
    
    for col, new_col in [
        ("has_hurricane_warning", "new_hurricane_warning"),
        ("has_hurricane_watch", "new_hurricane_watch"),
        ("has_tropical_storm_warning", "new_ts_warning"),
        ("has_tropical_storm_watch", "new_ts_watch"),
    ]:
        if col in df.columns:
            prev = df.groupby("storm")[col].shift(1).fillna(False)
            df[new_col] = (df[col] & ~prev).astype(int)
    df["rapid_intensification"] = (df["delta_wind_kt"] >= 35).astype(int)
    df["hours_since_prev_advisory"] = (
        df.groupby("storm")["utc_datetime"].diff().dt.total_seconds() / 3600)
    print(f"  Advisories: {len(df)}")
    return df

# storm phase assignment
def assign_phase(df_posts: pd.DataFrame, df_adv: pd.DataFrame) -> pd.Series:
    phase = pd.Series("active", index=df_posts.index)
    for storm in df_posts["storm"].unique():
        da = df_adv[df_adv["storm"] == storm]
        first, last = da["utc_datetime"].min(), da["utc_datetime"].max()
        mask = df_posts["storm"] == storm
        phase.loc[mask & (df_posts["created_utc"] < first)] = "pre_advisory"
        phase.loc[mask & (df_posts["created_utc"] > last)]  = "post_storm"
    return phase

# temporal merge
def merge_posts_advisories(df_posts: pd.DataFrame, df_adv: pd.DataFrame) -> pd.DataFrame:
    advisory_cols = [
        "utc_datetime","advisory_number","is_special","storm_type",
        "storm_category","max_wind_kt","min_pressure_mb",
        "movement_speed_kt","movement_direction","forecast_1_wind_kt",
        "has_hurricane_warning","has_hurricane_watch",
        "has_tropical_storm_warning","has_tropical_storm_watch",
        "delta_wind_kt","delta_pressure_mb","delta_category",
        "delta_forecast_1_wind","new_hurricane_warning","new_hurricane_watch",
        "new_ts_warning","new_ts_watch","rapid_intensification",
        "hours_since_prev_advisory",
    ]
    advisory_cols = [c for c in advisory_cols if c in df_adv.columns]

    df_posts["created_utc"] = df_posts["created_utc"].astype("datetime64[us, UTC]")
    df_adv["utc_datetime"]  = df_adv["utc_datetime"].astype("datetime64[us, UTC]")

    df_slim = (df_adv[["storm"] + advisory_cols]
               .rename(columns={"utc_datetime": "advisory_utc"})
               .sort_values("advisory_utc").reset_index(drop=True))

    df_merged = pd.merge_asof(
        df_posts.sort_values("created_utc").reset_index(drop=True),
        df_slim, left_on="created_utc", right_on="advisory_utc",
        by="storm", direction="backward",
    )
    df_merged["hours_since_advisory"] = (
        (df_merged["created_utc"] - df_merged["advisory_utc"])
        .dt.total_seconds() / 3600)
    return df_merged


# Main
def run() -> None:
    print("="*60)
    print("RPI Merge: Scored Posts <-> NHC Advisories")
    print("="*60)

    print("\nLoading advisories...")
    df_adv = prepare_advisories(ADVISORY_PATH)

    print("\nLoading scored posts...")
    dfs = []
    for storm in STORMS:
        path = SCORED_DIR / f"{storm}_scored.parquet"
        if not path.exists():
            print(f"  [skip] {path.name}")
            continue
        d = pd.read_parquet(path)
        dfs.append(d)
        print(f"  {storm}: {len(d)}")
    df_posts = pd.concat(dfs, ignore_index=True)
    df_posts["created_utc"] = pd.to_datetime(df_posts["created_utc"], utc=True)
    print(f"  Total: {len(df_posts)}")
 
    print("\nAssigning storm phases...")
    df_posts["storm_phase"] = assign_phase(df_posts, df_adv)
    for phase, n in df_posts["storm_phase"].value_counts().items():
        print(f"  {phase}: {n} ({100 * n / len(df_posts):.1f}%)")

    print("\nMerging...")
    df_merged = merge_posts_advisories(df_posts, df_adv)

    n_no_adv = df_merged["advisory_utc"].isna().sum()
    if n_no_adv:
        print(f"  Note: {n_no_adv} posts without advisory (pre-advisory)")

    summary = []
    for storm in STORMS:
        ds = df_merged[df_merged["storm"] == storm]
        active = ds[ds["storm_phase"] == "active"]
        if ds.empty: continue
        ds.to_parquet(OUTPUT_DIR / f"{storm}_merged.parquet", index=False)
        summary.append({
            "storm": storm,
            "n_total": len(ds),
            "n_active": len(active),
            "mean_RPI_composite": active["RPI_composite"].mean().round(4),
            "std_RPI_composite": active["RPI_composite"].std().round(4),
            "mean_RPI_composite_full": active["RPI_composite_full"].mean().round(4),
            "mean_RPI_A": active["RPI_A"].mean().round(4),
            "mean_RPI_A_log": active["RPI_A_log"].mean().round(4),
        })
        print(f"  {storm}: {len(ds)} posts saved")

    df_merged.to_parquet(OUTPUT_DIR / "all_storms_merged.parquet", index=False)

    df_sum = pd.DataFrame(summary)
    df_sum.to_csv(OUTPUT_DIR / "merge_summary.csv",
                  index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 60}\nMerge Summary:")
    print(df_sum.to_string(index=False))
    print(f"\nSaved to: {OUTPUT_DIR}")
 
 
if __name__ == "__main__":
    run()
