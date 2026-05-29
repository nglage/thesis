"""
NOT part of the main pipeline — run manually to verify that the NLI
relevance threshold choice (0.70) does not substantively affect corpus
composition.
 
Reported in thesis Section 4.2: corpus composition and RPI trajectories
were not substantively affected by the chosen threshold value.
 
Run from the project root:
  python src/data_processing/risk_score/sensitivity_check.py
"""

import pandas as pd
from paths import PATHS
from config.settings import STORMS
 
SCORED_DIR = PATHS["data_processed"] / "reddit" / "scored"
 
for storm in STORMS:
    df = pd.read_parquet(SCORED_DIR / f"{storm}_scored.parquet")
    for t in [0.60, 0.70, 0.80]:
        n = (df["p_hurricane_relevant"] >= t).sum()
        print(f"{storm} @ {t}: {n}/{len(df)} ({100*n/len(df):.1f}%)")