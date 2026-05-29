"""
- Run Steps 4 and 5: merge, descriptives, collinearity diagnostics, regression, time series plots
 
- Steps:
    - 4: rpi_merge -> merge scored posts with NHC advisories
    - 5a: descriptives -> summary statistics, validity checks, figures
    - 5b: predictor_diagnostics -> collinearity/VIF
    - 5c: regression -> fixed-effects OLS regression
    - 5d: timeseries -> time series visualisations
 
- Individual steps can also be run directly:
    - python src/data_processing/rpi_merge.py
    - python src/analysis/descriptives.py
    - python src/analysis/predictor_diagnostics.py
    - python src/analysis/regression.py
    - python src/analysis/timeseries.py
"""
 
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
 
from src.data_processing.risk_score.rpi_merge import run as run_merge
from src.analysis.descriptives import run as run_descriptives
from src.analysis.predictor_diagnostics import run as run_diagnostics
from src.analysis.regression import run as run_regression
from src.analysis.regression_diagnostics import run as run_reg_diagnostics
from src.analysis.timeseries import run as run_timeseries
 
if __name__ == "__main__":
    print("=" * 60)
    print("Step 4: Merge scored posts with NHC advisories")
    print("=" * 60)
    run_merge()
 
    print("\n" + "=" * 60)
    print("Step 5a: descriptive analysis")
    print("=" * 60)
    run_descriptives()
 
    print("\n" + "=" * 60)
    print("Step 5b: Predictor diagnostics (collinearity / VIF)")
    print("=" * 60)
    run_diagnostics()
 
    print("\n" + "=" * 60)
    print("Step 5c: Regression analysis")
    print("=" * 60)
    run_regression()


    print("\n" + "=" * 60)
    print("Step 5c.2: Regression diagnostics")
    print("=" * 60)
    run_reg_diagnostics()
 
    print("\n" + "=" * 60)
    print("Step 5d — Time series plots")
    print("=" * 60)
    run_timeseries()