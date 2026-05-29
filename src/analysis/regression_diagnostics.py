"""
- OLS regression diagnostics for the primary model
 
- Replicates R's 4-panel plot(lm) in Python:
     1. Residuals vs Fitted: linearity check
     2. Normal Q-Q: normality of residuals
     3. Scale-Location: homoscedasticity check
     4. Residuals vs Leverage: influential observations (Cook's distance)
 
- Also prints formal tests:
     - Breusch-Pagan test: homoscedasticity
     - Jarque-Bera test: normality
     - Durbin-Watson statistic: autocorrelation
 
- Run from project root:
     - python src/analysis/regression_diagnostics.py
 
Output: figures/regression/diagnostics_primary.png
        tables/regression_diagnostics.csv
"""

import warnings
warnings.filterwarnings("ignore")
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy import stats
from scipy.stats import jarque_bera
 
try:
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.stattools import durbin_watson
    HAS_SM = True
except ImportError:
    print("Install statsmodels: pip install statsmodels")
    HAS_SM = False
 
from src.analysis.data_prep import load_merged, add_features, build_hourly
from config.settings import ADVISORY_PREDS
from paths import PATHS
 

# Paths
FIGURE_DIR = PATHS["figures"] / "regression"
TABLE_DIR  = PATHS["tables"]
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True,  exist_ok=True)

# Fit primary model
def fit_primary_model(df_h: pd.DataFrame):
    """Fit the primary OLS model and return result + design matrix."""
    storm_cols = [c for c in df_h.columns
                  if c.startswith("storm_") and
                  c not in {"storm_phase", "storm_category", "storm_category_z",
                             "storm_hour_id", "storm_type"}]
    ctrl = ["storm_category_z", "log_n_posts", "hour_sin", "hour_cos"] + storm_cols
    preds = [p for p in ADVISORY_PREDS + ctrl
             if p in df_h.columns and df_h[p].notna().any()]
 
    df_m = df_h[["mean_rpi_composite"] + preds + ["adv_cycle_id"]].dropna()
    X = sm.add_constant(df_m[preds].astype(float))
    y = df_m["mean_rpi_composite"].astype(float)
 
    result = sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": df_m["adv_cycle_id"]},
    )
    print(f"Primary model: N={int(result.nobs)}  R²={result.rsquared:.4f}")
    return result, X, y

# 4-panel diagnostic plot
def plot_diagnostics(result, y: pd.Series, title_suffix: str = "",
                     filename: str = "diagnostics_primary.png") -> None:
    fitted     = result.fittedvalues
    residuals  = result.resid
    std_resid  = residuals / residuals.std()
    sqrt_std   = np.sqrt(np.abs(std_resid))
 
    # Influence measures
    influence  = result.get_influence()
    leverage   = influence.hat_matrix_diag
    cooks_d    = influence.cooks_distance[0]
 
    n          = len(residuals)
    top_idx    = np.argsort(np.abs(residuals))[-3:]   # label 3 most extreme
 
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"OLS Regression Diagnostics — Primary Model{' ' + title_suffix if title_suffix else ''}",
        fontsize=13, fontweight="bold",
    )

    # Residuals vs. fitted
    ax = axes[0, 0]
    ax.scatter(fitted, residuals, alpha=0.4, s=20, color="#457B9D", edgecolors="none")
    ax.axhline(0, color="red", lw=1.2, linestyle="--")

    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        lo = lowess(residuals, fitted, frac=0.5)
        ax.plot(lo[:, 0], lo[:, 1], color="red", lw=1.5)
    except Exception:
        pass
    for i in top_idx:
        ax.annotate(str(i), (fitted.iloc[i], residuals.iloc[i]),
                    fontsize=7, color="grey")
    ax.set_xlabel("Fitted values", fontsize=10)
    ax.set_ylabel("Residuals", fontsize=10)
    ax.set_title("Residuals vs Fitted", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)

    # Normal Q-Q
    ax = axes[0, 1]
    (osm, osr), (slope, intercept, r) = stats.probplot(std_resid, dist="norm")
    ax.scatter(osm, osr, alpha=0.5, s=20, color="#457B9D", edgecolors="none")
    xl = np.array([osm.min(), osm.max()])
    ax.plot(xl, slope * xl + intercept, color="red", lw=1.5, linestyle="--")
    # Label extremes
    for i in np.argsort(np.abs(osr))[-3:]:
        ax.annotate(str(i), (osm[i], osr[i]), fontsize=7, color="grey")
    ax.set_xlabel("Theoretical Quantiles", fontsize=10)
    ax.set_ylabel("Standardized residuals", fontsize=10)
    ax.set_title("Normal Q-Q", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)

    # Scale-location
    ax = axes[1, 0]
    ax.scatter(fitted, sqrt_std, alpha=0.4, s=20, color="#457B9D", edgecolors="none")
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        lo = lowess(sqrt_std, fitted, frac=0.5)
        ax.plot(lo[:, 0], lo[:, 1], color="red", lw=1.5)
    except Exception:
        pass
    ax.set_xlabel("Fitted values", fontsize=10)
    ax.set_ylabel("√|Standardized residuals|", fontsize=10)
    ax.set_title("Scale-Location", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)

    # Residuals vs. leverage
    ax = axes[1, 1]
    ax.scatter(leverage, std_resid, alpha=0.4, s=20, color="#457B9D", edgecolors="none")
    ax.axhline(0, color="grey", lw=0.8, linestyle="--")
 
    # Cook's distance contours (0.5 and 1.0)
    x_range = np.linspace(leverage.min(), leverage.max(), 200)
    p       = result.df_model + 1
    for cd_level, ls in [(0.5, "--"), (1.0, "-")]:
        y_cook = np.sqrt(cd_level * p * (1 - x_range) / x_range)
        ax.plot(x_range, y_cook,  color="red", lw=1.0, linestyle=ls, alpha=0.7)
        ax.plot(x_range, -y_cook, color="red", lw=1.0, linestyle=ls, alpha=0.7)
 
    for i in np.argsort(cooks_d)[-3:]:
        ax.annotate(str(i), (leverage[i], std_resid.iloc[i]),
                    fontsize=7, color="grey")
 
    cook_line = mlines.Line2D([], [], color="red", linestyle="--",
                               label="Cook's distance")
    ax.legend(handles=[cook_line], fontsize=8)
    ax.set_xlabel("Leverage", fontsize=10)
    ax.set_ylabel("Standardized residuals", fontsize=10)
    ax.set_title("Residuals vs Leverage", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
 
    plt.tight_layout()
    out = FIGURE_DIR / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out.name}")


# Formal diagnostics tests
def run_formal_tests(result, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Breusch-Pagan (homoscedasticity), Jarque-Bera (normality),
    Durbin-Watson (autocorrelation).
    """
    residuals = result.resid
    rows      = []
 
    # Breusch-Pagan
    try:
        bp_lm, bp_p, bp_f, bp_fp = het_breuschpagan(residuals, X)
        rows.append({
            "test":      "Breusch-Pagan",
            "statistic": round(bp_lm, 4),
            "p_value":   round(bp_p, 4),
            "null":      "Homoscedasticity",
            "result":    "Reject H0 (heteroscedastic)" if bp_p < 0.05
                         else "Fail to reject H0 (homoscedastic)",
        })
        print(f"Breusch-Pagan: LM={bp_lm:.4f}  p={bp_p:.4f}  "
              f"→ {'heteroscedastic' if bp_p < 0.05 else 'homoscedastic'}")
    except Exception as e:
        print(f"Breusch-Pagan failed: {e}")
 
    # Jarque-Bera
    jb_stat, jb_p = jarque_bera(residuals)
    rows.append({
        "test":      "Jarque-Bera",
        "statistic": round(jb_stat, 4),
        "p_value":   round(jb_p, 4),
        "null":      "Normality of residuals",
        "result":    "Reject H0 (non-normal)" if jb_p < 0.05
                     else "Fail to reject H0 (normal)",
    })
    print(f"Jarque-Bera:   JB={jb_stat:.4f}  p={jb_p:.4f}  "
          f"→ {'non-normal' if jb_p < 0.05 else 'normal'}")
 
    # Durbin-Watson
    dw = durbin_watson(residuals)
    rows.append({
        "test":      "Durbin-Watson",
        "statistic": round(dw, 4),
        "p_value":   None,
        "null":      "No autocorrelation (DW ≈ 2)",
        "result":    "Possible positive autocorrelation" if dw < 1.5
                     else "Possible negative autocorrelation" if dw > 2.5
                     else "No strong autocorrelation",
    })
    print(f"Durbin-Watson: DW={dw:.4f}  "
          f"→ {'positive autocorrelation' if dw < 1.5 else 'negative autocorrelation' if dw > 2.5 else 'no strong autocorrelation'}")
 
    df_tests = pd.DataFrame(rows)
    df_tests.to_csv(TABLE_DIR / "regression_diagnostics.csv",
                    index=False, encoding="utf-8-sig")
    return df_tests

# Main
def run() -> None:
    print("=" * 60)
    print("Regression Diagnostics")
    print("=" * 60)
    if not HAS_SM:
        return
 
    df_raw = load_merged()
    df     = add_features(df_raw, active_only=True)
    df_h   = build_hourly(df, label="full sample")
 
    print("\nFitting primary model...")
    result, X, y = fit_primary_model(df_h)
 
    print("\nGenerating 4-panel diagnostic plot...")
    plot_diagnostics(result, y)
 
    print("\nRunning formal tests...")
    run_formal_tests(result, X, y)
 
    print(f"\nDone.\n  Figure: {FIGURE_DIR / 'diagnostics_primary.png'}"
          f"\n  Table:  {TABLE_DIR / 'regression_diagnostics.csv'}")
 
 
if __name__ == "__main__":
    run()