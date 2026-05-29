"""
- Run Step 3: NLI + emotion scoring -> RPI components and composite.
 
- Requires a GPU for reasonable runtime
 
- Output:
    - data/processed/reddit/scored/{Storm}_scored.parquet
    - data/processed/reddit/scored/all_storms_scored.parquet
    - data/processed/reddit/scored/rpi_weights.json
"""
 
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
 
from src.data_processing.risk_score.rpi_scoring import run as run_scoring
 
if __name__ == "__main__":
    print("=" * 60)
    print("Step 3: RPI Scoring")
    print("=" * 60)
    run_scoring()
 