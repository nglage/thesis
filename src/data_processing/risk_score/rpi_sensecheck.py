"""
NOT part of the main pipeline — run manually to validate NLI scoring quality.
 
Exports stratified samples of posts for manual inspection:
  - High / medium / low RPI_composite
  - High scores on each of the three RPI dimensions and individual NLI labels
  - Random sample per storm
 
Each exported CSV contains the post text alongside all NLI scores so you
can read posts and verify the labels are sensible.
 
Also prints summary stats and the top/bottom posts per storm.

The "not selected" labels (p_uncertainty, p_protective) are included deliberately to confirm that the selected labels (p_likelihood, p_vulnerable)
are better operationalisations of the deliberative and experiential dimensions.

Run from the project root:
  python src/data_processing/risk_score/rpi_sensecheck.py
 
Output: data/processed/reddit/scored/sensecheck/
"""
 
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
 
import pandas as pd
import numpy as np
 
from paths import PATHS
from config.settings import STORMS
 
SCORED_DIR  = PATHS["data_processed"] / "reddit" / "scored"
OUTPUT_DIR  = SCORED_DIR / "sensecheck"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
 
SCORE_COLS = [
    "RPI_composite", "RPI_A", "RPI_D_rev", "RPI_E_rev",
    "fear", "p_fear_discourse",
    "p_likelihood", "p_vulnerable",
    "p_uncertainty", "p_protective", "p_factual",
    "p_hurricane_relevant",
]
 
DISPLAY_COLS = [
    "storm", "subreddit", "type", "created_utc",
    "text_for_scoring",
] + SCORE_COLS
 
 
def _load_storm(storm: str) -> pd.DataFrame | None:
    path = SCORED_DIR / f"{storm}_scored.parquet"
    if not path.exists():
        print(f"  [skip] {storm}: file not found")
        return None
    df = pd.read_parquet(path)
    df["storm"] = storm
    keep = [c for c in DISPLAY_COLS if c in df.columns]
    return df[keep].copy()
 
 
def _summary(df: pd.DataFrame, storm: str) -> None:
    print(f"\n{'─'*50}")
    print(f"{storm}  (N={len(df)})")
    for col in ["RPI_composite", "RPI_A", "RPI_D_rev", "RPI_E_rev"]:
        if col in df.columns:
            print(f"  {col:25s}: μ={df[col].mean():.3f}  σ={df[col].std():.3f}  "
                  f"p25={df[col].quantile(0.25):.3f}  p75={df[col].quantile(0.75):.3f}")
 
 
def _print_examples(df: pd.DataFrame, col: str, n: int = 3,
                     label: str = "") -> None:
    if col not in df.columns:
        return
    print(f"\n  Top {n} by {col} [{label}]:")
    top = df.nlargest(n, col)
    for _, row in top.iterrows():
        scores = "  |  ".join(
            f"{c}={row[c]:.3f}" for c in [
                "RPI_composite", "fear", "p_likelihood", "p_vulnerable",
                "p_hurricane_relevant",
            ]
            if c in row.index
        )
        print(f"    [{scores}]")
        print(f"    {str(row.get('text_for_scoring', ''))[:200]}")
 
 
def _stratified_sample(df: pd.DataFrame, storm: str,
                        n_per_stratum: int = 15) -> pd.DataFrame:
    """
    Sample posts from each RPI tercile and each key label stratum.
    Includes both selected labels (used in RPI) and non-selected labels
    (excluded during validation) to support comparative inspection.
    """
    samples = []
 
    # RPI composite terciles
    if "RPI_composite" in df.columns:
        df["rpi_tercile"] = pd.qcut(df["RPI_composite"], 3,
                                     labels=["low", "mid", "high"],
                                     duplicates="drop")
        for tercile in ["low", "mid", "high"]:
            subset = df[df["rpi_tercile"] == tercile]
            s = subset.sample(min(n_per_stratum, len(subset)), random_state=42).copy()
            s["stratum"] = f"RPI_{tercile}"
            samples.append(s)
 
    # Selected dimension labels
    for col, label in [
        ("fear",         "high_affective_fear"),        # RPI_A
        ("p_likelihood", "high_deliberative"),           # RPI_D
        ("p_vulnerable", "high_experiential"),           # RPI_E
    ]:
        if col not in df.columns:
            continue
        subset = df.nlargest(max(n_per_stratum * 3, 50), col)
        s = subset.sample(min(n_per_stratum, len(subset)), random_state=42).copy()
        s["stratum"] = label
        samples.append(s)
 
    # Non-selected labels — included to validate exclusion decision
    for col, label in [
        ("p_uncertainty", "high_uncertainty_not_selected"),
        ("p_protective",  "high_protective_not_selected"),
        ("p_factual",     "high_factual"),
        ("p_fear_discourse", "high_nli_fear_discourse"),
    ]:
        if col not in df.columns:
            continue
        subset = df.nlargest(max(n_per_stratum * 3, 50), col)
        s = subset.sample(min(n_per_stratum, len(subset)), random_state=42).copy()
        s["stratum"] = label
        samples.append(s)
 
    # Random baseline
    s = df.sample(min(n_per_stratum, len(df)), random_state=42).copy()
    s["stratum"] = "random"
    samples.append(s)
 
    combined = pd.concat(samples, ignore_index=True)
    combined["storm"] = storm
    combined = combined.drop(columns=["rpi_tercile"], errors="ignore")
    return combined
 
 
def run() -> None:
    print("=" * 60)
    print("RPI Sensecheck — Manual Validation Sample")
    print("=" * 60)
 
    all_samples = []
 
    for storm in STORMS:
        df = _load_storm(storm)
        if df is None:
            continue
 
        _summary(df, storm)
 
        # RPI dimensions (selected labels)
        _print_examples(df, "RPI_composite",    n=3, label="highest RPI_composite")
        _print_examples(df, "fear",             n=3, label="highest fear / RPI_A (affective)")
        _print_examples(df, "p_likelihood",     n=3, label="highest p_likelihood / RPI_D (deliberative)")
        _print_examples(df, "p_vulnerable",     n=3, label="highest p_vulnerable / RPI_E (experiential)")
 
        # Non-selected labels — validate exclusion
        _print_examples(df, "p_uncertainty",    n=3, label="highest p_uncertainty (not selected)")
        _print_examples(df, "p_protective",     n=3, label="highest p_protective (not selected)")
        _print_examples(df, "p_factual",        n=3, label="highest p_factual")
        _print_examples(df, "p_fear_discourse", n=3, label="highest NLI fear discourse")
 
        # Per-storm stratified sample to CSV
        sample = _stratified_sample(df, storm, n_per_stratum=15)
        out    = OUTPUT_DIR / f"{storm}_sensecheck.csv"
        sample.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n  Saved {len(sample)} rows → {out.name}")
        all_samples.append(sample)
 
    # Combined file
    if all_samples:
        df_all = pd.concat(all_samples, ignore_index=True)
        out    = OUTPUT_DIR / "ALL_storms_sensecheck.csv"
        df_all.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\nCombined: {len(df_all)} rows → {out.name}")
 
    print(f"\nDone. Open CSVs in Excel to read posts and verify labels.")
    print(f"  Output: {OUTPUT_DIR}")
 
 
if __name__ == "__main__":
    run()