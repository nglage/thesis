"""
Shared data preparation module
 
Imported by descriptives.py, regression.py, timeseries.py, and
predictor_diagnostics.py.  Provides:
 
  load_merged() -> load all_storms_merged.parquet
  load_advisories() -> load hurricane_forecast_advisories.csv
  add_features() -> subreddit types, time features, lag variables, z-standardised predictors, storm/subreddit dummies
  filter_proximity() -> restrict to state + local community only
  build_hourly() -> aggregate to storm-hour level
  smooth() -> rolling mean helper for time series plots
  describe_sample() -> quick print summary
"""

from pathlib import Path
import numpy as np
import pandas as pd

from config.settings import (
    STORMS, STORM_COLORS, SUBREDDIT_TYPE_MAP, PROXIMITY_TYPES,
    OUTCOME_COLS, MIN_POSTS_PER_HOUR, ROLLING_WINDOW, ADVISORY_PREDS,
)
from paths import PATHS

# Advisory variables kept at hourly level (constant within advisory cycle)
ADVISORY_VARS = [
    "storm", "hour_floor", "advisory_number",
    "delta_wind_kt_z", "delta_forecast_1_wind_z",
    "new_hurricane_warning", "new_hurricane_watch",
    "delta_wind_lag1_z", "delta_fc_lag1_z", "warning_lag1",
    "storm_category_z", "max_wind_kt",
    "max_wind_kt_z",
    # Raw wind variables for timeseries plots
    "max_wind_kt", "forecast_1_wind_kt",
]

# load functions
def load_merged(path: Path | None = None) -> pd.DataFrame:
    path = path or PATHS["data_processed"] / "analysis" / "all_storms_merged.parquet"
    df = pd.read_parquet(path)
    df["created_utc"] = pd.to_datetime(df["created_utc"],  utc=True)
    df["advisory_utc"] = pd.to_datetime(df["advisory_utc"], utc=True)
    print(f"Loaded merged data: {len(df)} posts")
    return df


def load_advisories(path: Path | None = None) -> pd.DataFrame:
    path = path or PATHS["data_processed"] / "hurricane_data" / "hurricane_forecast_advisories.csv"
    adv = pd.read_csv(path)
    adv["utc_datetime"] = pd.to_datetime(adv["utc_datetime"], utc=True)
    adv = adv.sort_values(["storm", "utc_datetime"]).reset_index(drop=True)
    print(f"Loaded advisories: {len(adv)} rows")
    return adv

# feature engineering
def add_features(df: pd.DataFrame, active_only: bool = True) -> pd.DataFrame:
    """
    Add all derived features needed for regression and timeseries:
      - subreddit_type and is_proximity flag
      - is_submission flag
      - hour_floor, hour_of_day, hour_sin, hour_cos, storm_hour_id
      - lag variables (delta_wind_lag1, delta_fc_lag1, warning_lag1)
      - z-standardised continuous predictors
      - storm fixed effect dummies
      - subreddit type dummies

    active_only: restrict to storm_phase == 'active' and advisory_utc not null
    """
    if active_only:
        df = df[df["storm_phase"] == "active"].copy()
        df = df.dropna(subset=["advisory_utc"]).copy()
        print(f"Active posts with advisory: {len(df)}")

    # Subreddit classification
    df["subreddit_type"] = (df["subreddit"]
                            .map(SUBREDDIT_TYPE_MAP)
                            .fillna("general_weather"))
    df["is_proximity"] = df["subreddit_type"].isin(PROXIMITY_TYPES)
    df["is_submission"] = (df["type"] == "submission").astype(int)

    # Time features
    df["hour_floor"] = df["created_utc"].dt.floor("h")
    df["hour_of_day"] = df["created_utc"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["storm_hour_id"] = df["storm"] + "_" + df["hour_floor"].astype(str)

    # Lag variables (computed from unique advisory rows)
    adv_lags = (
        df[["storm", "advisory_number", "delta_wind_kt",
            "delta_forecast_1_wind", "new_hurricane_warning"]]
        .drop_duplicates(subset=["storm", "advisory_number"])
        .sort_values(["storm", "advisory_number"]).copy()
    )
    adv_lags["delta_wind_lag1"] = (
        adv_lags.groupby("storm")["delta_wind_kt"].shift(1))
    adv_lags["delta_fc_lag1"] = (
        adv_lags.groupby("storm")["delta_forecast_1_wind"].shift(1))
    adv_lags["warning_lag1"] = (
        adv_lags.groupby("storm")["new_hurricane_warning"].shift(1))

    df = df.merge(
        adv_lags[["storm", "advisory_number",
                  "delta_wind_lag1", "delta_fc_lag1", "warning_lag1"]],
        on=["storm", "advisory_number"], how="left")

    # Standardise continuous predictors (standardise on FULL dataset)
    for var in ["delta_wind_kt", "delta_forecast_1_wind",
                "delta_wind_lag1", "delta_fc_lag1",
                "storm_category", "max_wind_kt"]:
        if var in df.columns:
            mu = df[var].mean()
            sd = df[var].std()
            df[f"{var}_z"] = (df[var] - mu) / (sd + 1e-8)

    # Binary warning variables -> integer
    for col in ["new_hurricane_warning", "new_hurricane_watch", "warning_lag1"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Storm fixed effect dummies
    storm_dummies = pd.get_dummies(df["storm"], prefix="storm",
                                    drop_first=True).astype(int)
    df = pd.concat([df, storm_dummies], axis=1)

    # Subreddit type dummies (general_weather = reference)
    sub_dummies = pd.get_dummies(df["subreddit_type"], prefix="sub",
                                  drop_first=False).astype(int)
    sub_dummies = sub_dummies.drop(columns=["sub_general_weather"], errors="ignore")
    df = pd.concat([df, sub_dummies], axis=1)

    return df


# Filtering 
def filter_proximity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Restrict to state and local community subreddits only.
    Excludes general weather communities (r/hurricane, r/TropicalWeather).

    Note: subreddit_type must already be set via add_features().
    """
    df_prx = df[df["is_proximity"]].copy()
    n_total = len(df_prx)
    counts  = df_prx["subreddit_type"].value_counts().to_dict()
    print(f"\nProximity-restricted subsample: {n_total} posts")
    for sub_type, n in counts.items():
        print(f"  {sub_type}: {n} ({100*n/n_total:.1f}%)")
    return df_prx


# Hourly aggregation
def build_hourly(df: pd.DataFrame,
    label: str = "",
    min_posts: int = MIN_POSTS_PER_HOUR,) -> pd.DataFrame:
    """
    Aggregate post-level data to storm-hour level

    Returns hourly dataframe with:
      - RPI outcome means
      - advisory variables (first value within hour -> constant within cycle)
      - storm fixed effect dummies
      - log_n_posts
      - adv_cycle_id for clustering standard errors
    """
    # Outcome aggregation — only include columns that exist
    agg_dict = {k: v for k, v in OUTCOME_COLS.items() if v[0] in df.columns}

    hourly = df.groupby(["storm", "hour_floor"]).agg(**agg_dict).reset_index()

    # Advisory variables: constant within advisory cycle, take first value
    adv_vars = [v for v in ADVISORY_VARS if v in df.columns]
    adv_vars = list(dict.fromkeys(["storm", "hour_floor"] + adv_vars))
    adv_hourly = (
        df[adv_vars]
        .groupby(["storm", "hour_floor"]).first()
        .reset_index()
    )
    hourly = hourly.merge(adv_hourly, on=["storm", "hour_floor"], how="left")

    # Advisory cycle cluster ID
    hourly["adv_cycle_id"] = (hourly["storm"] + "_" +
                               hourly["advisory_number"].astype(str))

    # Standardise max_wind_kt at hourly level
    if "max_wind_kt" in hourly.columns:
        mu = hourly["max_wind_kt"].mean()
        sd = hourly["max_wind_kt"].std()
        hourly["max_wind_kt_z"] = (hourly["max_wind_kt"] - mu) / (sd + 1e-8)

    # Storm FE dummies
    storm_dummies = pd.get_dummies(hourly["storm"], prefix="storm",
                                    drop_first=True).astype(int)
    hourly = pd.concat([hourly, storm_dummies], axis=1)

    # Log post volume
    hourly["log_n_posts"] = np.log(hourly["n_posts"] + 1)

    # Filter sparse hours
    hourly = hourly[hourly["n_posts"] >= min_posts].copy()

    tag = f" [{label}]" if label else ""
    n_clusters = hourly["adv_cycle_id"].nunique()
    print(f"\nHourly dataset{tag}: {len(hourly)} storm-hours  "
          f"({n_clusters} advisory-cycle clusters)")
    return hourly


# Smoothing function (for timeseries plots)
def smooth(hourly: pd.DataFrame, col: str, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Apply backward-looking rolling mean. Returns a two-column dataframe."""
    return (
        hourly.set_index("hour_floor")[col]
        .rolling(window=window, min_periods=3)
        .mean()
        .reset_index()
    )


# summary
def describe_sample(df: pd.DataFrame, label: str = "") -> None:
    tag = f" [{label}]" if label else ""
    print(f"\nSample summary{tag}:")
    print(f"  Total posts: {len(df)}")
    for storm in STORMS:
        n = (df["storm"] == storm).sum()
        print(f"  {storm}: {n}")
    if "subreddit_type" in df.columns:
        print("\n  By subreddit type:")
        for st, n in df["subreddit_type"].value_counts().items():
            print(f"    {st}: {n} ({100 * n / len(df):.1f}%)")