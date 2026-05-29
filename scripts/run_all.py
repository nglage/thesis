"""
- Run the complete pipeline end-to-end.

  - Step 1a: scrape_nhc_archive -> hurricane_archive_links.csv
  - Step 1b: scrape_forecast_advisories -> hurricane_forecast_advisories.csv

  - Step 2a: reddit_preprocessing -> {Storm}_*.parquet  (time-filtered)
  - Step 2b: reddit_filtering -> {Storm}_*_filtered.parquet

  - Step 3: rpi_scoring -> {Storm}_scored.parquet + all_storms_scored.parquet

  - Step 4: rpi_merge -> {Storm}_merged.parquet + all_storms_merged.parquet
  - Step 5a: descriptives -> figures/descriptives/ + tables/descriptives/
  - Step 5b: predictor_diagnostics -> figures/regression/predictor_* + tables/predictor_*
  - Step 5c: regression -> figures/regression/ + tables/
  - Step 5d: timeseries -> figures/timeseries/

- Usage:
  - python scripts/run_all.py

- Skip data collection/preprocessing:
  - python scripts/run_all.py --skip-nhc
  - python scripts/run_all.py --skip-reddit
  - python scripts/run_all.py --skip-scoring

- Skip to analysis only:
  - python scripts/run_all.py --analysis-only
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_processing.hurricane_data.scrape_nhc_archive import run as run_archive
from src.data_processing.hurricane_data.scrape_forecast_advisories import run as run_advisories
from src.data_processing.reddit_data.reddit_preprocessing import run as run_preprocess
from src.data_processing.reddit_data.reddit_filtering import run as run_filter
from src.data_processing.risk_score.rpi_scoring import run as run_scoring
from src.data_processing.risk_score.rpi_merge import run as run_merge
from src.analysis.descriptives import run as run_descriptives
from src.analysis.predictor_diagnostics import run as run_diagnostics
from src.analysis.regression_diagnostics import run as run_reg_diagnostics
from src.analysis.regression import run as run_regression
from src.analysis.timeseries import run as run_timeseries


def _header(text: str) -> None:
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full hurricane RPI pipeline.")
    parser.add_argument("--skip-nhc", action="store_true",
                        help="Skip NHC scraping (Steps 1a/1b)")
    parser.add_argument("--skip-reddit", action="store_true",
                        help="Skip Reddit preprocessing (Steps 2a/2b)")
    parser.add_argument("--skip-scoring", action="store_true",
                        help="Skip RPI scoring (Step 3)")
    parser.add_argument("--analysis-only", action="store_true",
                        help="Run Steps 4–5 only (merge + analysis)")
    parser.add_argument("--no-analysis", action="store_true",
                    help="Run only through Step 4 (no analysis)")
    args = parser.parse_args()

    if args.analysis_only:
        args.skip_nhc = args.skip_reddit = args.skip_scoring = True

    if not args.skip_nhc:
        _header("Step 1a: NHC archive scrape")
        run_archive()
        _header("Step 1b: Forecast advisory parsing")
        run_advisories()

    if not args.skip_reddit:
        _header("Step 2a: Reddit time-filtering")
        run_preprocess()
        _header("Step 2b: Reddit keyword filtering + cleaning")
        run_filter()

    if not args.skip_scoring:
        _header("Step 3: RPI scoring  (NLI + emotion classifier)")
        run_scoring()

    _header("Step 4: Merge scored posts with NHC advisories")
    run_merge()

    if not args.no_analysis:
        pass
        _header("Step 5a: Descriptive analysis")
        run_descriptives()

        _header("Step 5b: Predictor diagnostics  (collinearity / VIF)")
        run_diagnostics()

        _header("Step 5c: Regression analysis")
        run_regression()

        _header("Step 5c.2: Regression diagnostics")
        run_reg_diagnostics()

        _header("Step 5d: Time series plots")
        run_timeseries()

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()