"""
- Step 5b: Collinearity and VIF diagnostics
 
- Computes pairwise correlation matrix and Variance Inflation Factors for all candidate regression predictors
- Used before model estimation, confirm that raw wind level variables are near-perfectly collinear -> justify exclusion from primary specification
 
Output: figures/regression/predictor_correlation_heatmap.png
        figures/regression/predictor_vif.png
        tables/predictor_correlations.csv
        tables/predictor_vif.csv
        tables/high_correlations.csv
        tables/predictor_outcome_correlations.csv
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats as scipy_stats
 
try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_SM = True
except ImportError:
    print("Install statsmodels: pip install statsmodels")
    HAS_SM = False
 
from src.analysis.data_prep import load_merged, add_features, build_hourly
from paths import PATHS

# Paths
FIGURE_DIR = PATHS["figures"] / "regression"
TABLE_DIR  = PATHS["tables"]
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# Load and prepare hourly dataset
def _load_hourly_diagnostics() -> pd.DataFrame:
    df = load_merged()
    df = add_features(df, active_only=True)
    df["hour_floor"]  = df["created_utc"].dt.floor("h")

    # Compute lag variables + raw level variables
    adv_base = (
        df[["storm", "advisory_number", "delta_wind_kt",
            "delta_forecast_1_wind", "new_hurricane_warning",
            "max_wind_kt", "forecast_1_wind_kt"]]
        .drop_duplicates(subset=["storm", "advisory_number"])
        .sort_values(["storm", "advisory_number"]).copy()
    )

    # Lag1 of all advisory variables
    for col in ["delta_wind_kt", "delta_forecast_1_wind",
                "max_wind_kt", "forecast_1_wind_kt",
                "new_hurricane_warning"]:
        adv_base[f"{col}_lag1"] = adv_base.groupby("storm")[col].shift(1)

    df = df.merge(
        adv_base[["storm", "advisory_number",
                  "delta_wind_kt_lag1", "delta_forecast_1_wind_lag1",
                  "max_wind_kt_lag1", "forecast_1_wind_kt_lag1",
                  "new_hurricane_warning_lag1"]],
        on=["storm", "advisory_number"], how="left"
    ).rename(columns={
        "delta_wind_kt_lag1":          "delta_wind_lag1",
        "delta_forecast_1_wind_lag1":  "delta_fc_lag1",
        "max_wind_kt_lag1":            "max_wind_lag1",
        "forecast_1_wind_kt_lag1":     "forecast_lag1_wind_kt",
        "new_hurricane_warning_lag1":  "warning_lag1",
    })
    
    # Remove duplicate columns from merge
    df = df.loc[:, ~df.columns.duplicated()]

    for var in ["delta_wind_kt", "delta_forecast_1_wind", "delta_wind_lag1", "delta_fc_lag1",
                "max_wind_lag1", "forecast_lag1_wind_kt", "storm_category"]:
        if var in df.columns:
            mu = df[var].mean(); sd = df[var].std()
            df[f"{var}_z"] = (df[var] - mu) / (sd + 1e-8)
 
    for col in ["new_hurricane_warning", "new_hurricane_watch", "warning_lag1"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
 
    hourly = build_hourly(df, label="diagnostics")
    return hourly


# Correlation matrix
PRED_GROUPS = {
    "Wind level": [
            "max_wind_kt_z",
            "max_wind_lag1_z",
        ],
        "Forecast level": [
            "forecast_1_wind_kt_z",
            "forecast_lag1_wind_kt_z",
        ],
        "Wind change": [
            "delta_wind_kt_z",
            "delta_wind_lag1_z",
        ],
        "Forecast change": [
            "delta_forecast_1_wind_z",
            "delta_fc_lag1_z",
        ],
        "Warning": [
            "new_hurricane_warning",
            "warning_lag1",
            "new_hurricane_watch",
        ],
        "Controls": [
            "storm_category_z",
            "log_n_posts",
            "hour_sin",
            "hour_cos",
        ],
    }
 
LABEL_MAP = {
    "max_wind_kt_z":           "Current wind\n(level)",
    "max_wind_lag1_z":         "Current wind\n(level, lag1)",
    "forecast_1_wind_kt_z":    "Forecast wind\n(level)",
    "forecast_lag1_wind_kt_z": "Forecast wind\n(level, lag1)",
    "delta_wind_kt_z":         "Current wind\n(change)",
    "delta_wind_lag1_z":       "Current wind\n(change, lag1)",
    "delta_forecast_1_wind_z": "Forecast wind\n(change)",
    "delta_fc_lag1_z":         "Forecast wind\n(change, lag1)",
    "new_hurricane_warning":   "Warning\n(new)",
    "warning_lag1":            "Warning\n(lag1)",
    "new_hurricane_watch":     "Watch\n(new)",
    "storm_category_z":        "Storm\ncategory",
    "log_n_posts":             "Log post\nvolume",
    "hour_sin":                "Hour\n(sin)",
    "hour_cos":                "Hour\n(cos)",
}
 

def compute_correlations(hourly: pd.DataFrame) -> tuple[pd.DataFrame, list]:

    # Flatten in group order
    all_preds = list(dict.fromkeys(
        p for gp in PRED_GROUPS.values() for p in gp if p in hourly.columns
    ))
    df_corr  = hourly[all_preds].dropna()
    corr     = df_corr.corr().round(3)
    print("\nCorrelation matrix (advisory predictors):")
    print(corr.to_string())
    corr.to_csv(TABLE_DIR / "predictor_correlations.csv", encoding="utf-8-sig")

    labels = [LABEL_MAP.get(p, p) for p in all_preds]

    # Heatmap
    fig, ax = plt.subplots(figsize=(13, 11))

    # Diverging colormap centred at 0
    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im   = ax.imshow(corr.values, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(all_preds)))
    ax.set_yticks(range(len(all_preds)))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=8)

    # Annotate cells
    for i in range(len(all_preds)):
        for j in range(len(all_preds)):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    # Group dividers
    group_sizes = [len([p for p in gp if p in hourly.columns]) for gp in PRED_GROUPS.values()]
    cumulative = 0
    for size in group_sizes[:-1]:
        cumulative += size
        ax.axhline(cumulative - 0.5, color="white", lw=2)
        ax.axvline(cumulative - 0.5, color="white", lw=2)

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04,
                 label="Pearson correlation")
    ax.set_title("Predictor Correlation Matrix\n"
                 "(hourly aggregated dataset, n={})".format(len(df_corr)),
                 fontsize=12, fontweight="bold", pad=15)

    plt.tight_layout()
    out = FIGURE_DIR / "predictor_correlation_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" Saved: predictor_correlation_heatmap.png")

    # Flag high correlations
    high = [
        {"var1": all_preds[i], "var2": all_preds[j], "r": round(corr.values[i, j], 3)}
        for i in range(len(all_preds))
        for j in range(i + 1, len(all_preds))
        if abs(corr.values[i, j]) > 0.7
    ]
    print("\nHigh correlations (|r| > 0.7):")
    for h in high:
        print(f"  {h['var1']} × {h['var2']}: r={h['r']}")
    if not high:
        print("  None above 0.7")
    pd.DataFrame(high).to_csv(TABLE_DIR / "high_correlations.csv",
                               index=False, encoding="utf-8-sig")
    return corr, all_preds


# VIF
def compute_vif(hourly: pd.DataFrame, all_preds: list) -> pd.DataFrame | None:
    """
    Compute Variance Inflation Factor for each predictor.
    VIF > 5 = moderate concern; VIF > 10 = serious concern.
    """
    if not HAS_SM:
        return None

    df_vif = hourly[all_preds].dropna()

    # Add constant for VIF computation
    X = sm.add_constant(df_vif.astype(float))

    vif_data = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        try:
            vif = variance_inflation_factor(X.values, i)
            vif_data.append({"variable": col, "VIF": round(vif, 3)})
        except Exception as e:
            vif_data.append({"variable": col, "VIF": np.nan})
            print(f"  VIF error for {col}: {e}")

    vif_df = pd.DataFrame(vif_data).sort_values("VIF", ascending=False)

    print("\nVIF:")
    print(vif_df.to_string(index=False))
    vif_df.to_csv(TABLE_DIR / "predictor_vif.csv", index=False, encoding="utf-8-sig")
 
    fig, ax = plt.subplots(figsize=(8, max(4, len(vif_df) * 0.4)))
    colors  = ["#E63946" if v > 10 else "#F4A261" if v > 5 else "#2A9D8F"
               for v in vif_df["VIF"]]
    ax.barh(range(len(vif_df)), vif_df["VIF"], color=colors, alpha=0.8)
    ax.axvline(5,  color="#F4A261", lw=1.5, linestyle="--", label="VIF = 5")
    ax.axvline(10, color="#E63946", lw=1.5, linestyle="--", label="VIF = 10")
    ax.set_yticks(range(len(vif_df)))
    ax.set_yticklabels(vif_df["variable"], fontsize=8)
    ax.set_xlabel("VIF", fontsize=10)
    ax.set_title("Variance Inflation Factors — Candidate Predictors",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "predictor_vif.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: predictor_vif.png")
    return vif_df


# Correlation with outcome
def correlate_with_outcome(hourly: pd.DataFrame, all_preds: list) -> pd.DataFrame:
    """
    Simple bivariate correlations of each predictor with
    mean_rpi_composite -> useful for a quick sanity check before running the full model
    """
    print("\nBivariate correlations with mean_rpi_composite:")
    rows = []
    for pred in all_preds:
        if pred not in hourly.columns:
            continue
        valid = hourly[[pred, "mean_rpi_composite"]].dropna()
        if len(valid) < 10:
            continue
        r, p = scipy_stats.pearsonr(valid[pred], valid["mean_rpi_composite"])
        sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.1 else ""
        print(f"  {pred:35s}: r={r:+.3f}  p={p:.4f} {sig}")
        rows.append({"predictor": pred, "r": round(r, 3), "p": round(p, 4), "sig": sig})
    outcome_corr = pd.DataFrame(rows).sort_values("r", key=abs, ascending=False)
    outcome_corr.to_csv(TABLE_DIR / "predictor_outcome_correlations.csv",
                        index=False, encoding="utf-8-sig")
    return outcome_corr


# ── Main ──────────────────────────────────────────────────────────────────────
def run() -> None:
    print("=" * 60)
    print("Predictor Correlation & VIF Diagnostics")
    print("=" * 60)
 
    hourly = _load_hourly_diagnostics()
 
    print("\n[1] Correlation matrix...")
    corr, all_preds = compute_correlations(hourly)
 
    print("\n[2] VIF...")
    compute_vif(hourly, all_preds)
 
    print("\n[3] Bivariate correlations with outcome...")
    correlate_with_outcome(hourly, all_preds)
 
    print(f"\n{'=' * 60}\nDone.\n  Tables:  {TABLE_DIR}\n  Figures: {FIGURE_DIR}")
 
 
if __name__ == "__main__":
    run()