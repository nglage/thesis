"""
NOT part of the main pipeline -> run manually to inspect scored data
 
Diagnostic checks run during development:
  - Basic summary stats per storm
  - Manual validity checks (top/bottom RPI posts)
  - Post volume and sparsity checks per hour
  - Phase-split volume check (pre / active / post-storm)
 
Run from the project root:
  python src/data_processing/risk_score/inspection.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
 
from paths import PATHS
from config.settings import STORMS
 
SCORED_DIR    = PATHS["data_processed"] / "reddit" / "scored"
ADVISORY_PATH = PATHS["data_processed"] / "hurricane_data" / "hurricane_archive_links.csv"


# Summary stats across storms
print("=" * 50)
print("Summary stats")
print("=" * 50)
 
for f in sorted(SCORED_DIR.glob("*_scored.parquet")):
    df = pd.read_parquet(f)
    print(f"\n{f.stem}")
    print(f"  N: {len(df)}")
    print(f"  Mean RPI_simple: {df['RPI_simple'].mean():.3f}")
    print(f"  Mean RPI_composite: {df['RPI_composite'].mean():.3f}")
    print(f"  % p_dismissive > 0.5: {(df['p_dismissive'] > 0.5).mean() * 100:.1f}%")
    print(f"  % p_protective > 0.5: {(df['p_protective'] > 0.5).mean() * 100:.1f}%")
    print(f"  Subreddits: {df['subreddit'].value_counts().to_dict()}")

# manual validity check - Harvey
print("\n" + "=" * 50)
print("Manual validity check — Harvey")
print("=" * 50)
 
df = pd.read_parquet(SCORED_DIR / "Harvey_scored.parquet")
 
print("\n=== TOP 10 HIGHEST RPI_simple ===")
top = df.nlargest(10, "RPI_simple")[["text_for_scoring", "RPI_simple", "p_dismissive"]]
for _, row in top.iterrows():
    print(f"\nRPI={row['RPI_simple']:.3f} | dismissive={row['p_dismissive']:.3f}")
    print(f"  {row['text_for_scoring'][:200]}")
 
print("\n=== TOP 10 LOWEST RPI_simple ===")
bottom = df.nsmallest(10, "RPI_simple")[["text_for_scoring", "RPI_simple", "p_dismissive"]]
for _, row in bottom.iterrows():
    print(f"\nRPI={row['RPI_simple']:.3f} | dismissive={row['p_dismissive']:.3f}")
    print(f"  {row['text_for_scoring'][:200]}")

# construct validity correlations
print("\n=== Correlations ===")
corr_cols = [c for c in ["RPI_simple", "RPI_composite", "RPI_D_rev", "RPI_E_rev"]
             if c in df.columns]
print(df[corr_cols].corr().round(3))

# quick temporal plot - Harvey
df["hour"] = df["created_utc"].dt.floor("h")
hourly     = df.groupby("hour")["RPI_simple"].mean()
 
plt.figure(figsize=(14, 4))
plt.plot(hourly.index, hourly.values)
plt.title("Harvey — Mean RPI_simple per hour")
plt.xlabel("Date")
plt.ylabel("Mean fear score")
plt.tight_layout()
plt.savefig(SCORED_DIR / "Harvey_RPI_timeseries.png", dpi=150)
plt.show()

# post volume per hour - check sparsity
print("\n" + "=" * 50)
print("Post volume per hour")
print("=" * 50)
 
for f in sorted(SCORED_DIR.glob("*_scored.parquet")):
    df = pd.read_parquet(f)
    df["hour"] = pd.to_datetime(df["created_utc"], utc=True).dt.floor("h")
    hourly_n = df.groupby("hour")["RPI_simple"].count()
    print(f"\n{f.stem.replace('_scored', '')}")
    print(f"  Total hours:       {len(hourly_n)}")
    print(f"  Median posts/hour: {hourly_n.median():.0f}")
    print(f"  Hours < 5 posts:   {(hourly_n < 5).sum()} ({100 * (hourly_n < 5).mean():.1f}%)")
    print(f"  Hours < 10 posts:  {(hourly_n < 10).sum()} ({100 * (hourly_n < 10).mean():.1f}%)")
    print(f"  Hours < 20 posts:  {(hourly_n < 20).sum()} ({100 * (hourly_n < 20).mean():.1f}%)")

# Split into pre-storm, active, post-storm
print("\n" + "=" * 50)
print("Volume by storm phase (pre / active / post)")
print("=" * 50)
 
advisories = pd.read_csv(ADVISORY_PATH)
advisories["utc_datetime"] = pd.to_datetime(advisories["utc_datetime"], utc=True)
 
for storm in STORMS:
    path = SCORED_DIR / f"{storm}_scored.parquet"
    if not path.exists():
        continue
 
    df = pd.read_parquet(path)
    df["hour"] = pd.to_datetime(df["created_utc"], utc=True).dt.floor("h")
    hourly_n = df.groupby("hour")["RPI_simple"].count().reset_index()
    hourly_n.columns = ["hour", "n_posts"]
 
    storm_adv = advisories[advisories["storm"] == storm]
    first_advisory = storm_adv["utc_datetime"].min()
    last_advisory = storm_adv["utc_datetime"].max()
 
    pre = hourly_n[hourly_n["hour"] < first_advisory]
    active = hourly_n[(hourly_n["hour"] >= first_advisory) &
                      (hourly_n["hour"] <= last_advisory)]
    post = hourly_n[hourly_n["hour"] > last_advisory]
 
    print(f"\n{storm} | first: {first_advisory.date()} | last: {last_advisory.date()}")
    print(f" Pre: {len(pre):3d} hrs | median: {pre['n_posts'].median():5.0f} | <10: {(pre['n_posts'] < 10).sum()}")
    print(f" Active: {len(active):3d} hrs | median: {active['n_posts'].median():5.0f} | <10: {(active['n_posts'] < 10).sum()}")
    print(f" Post: {len(post):3d} hrs | median: {post['n_posts'].median():5.0f} | <10: {(post['n_posts'] < 10).sum()}")
 