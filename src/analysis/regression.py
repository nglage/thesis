"""
Step 5c — Fixed-effects OLS regression analysis
 
Six analyses as reported in the thesis:
  1. Primary model (storm_category_z) + incremental R²
  2. Robustness A: RPI_composite_full as outcome
  3. Robustness B: max_wind_kt_z replacing storm_category_z
  4. Multi-outcome comparison + incremental R² for each
  5. Exploratory community proximity (post-level)
  6. Proximity-restricted subsample:
       6a. Primary model on proximity posts only
       6b. Sub-score regressions (RPI_A, RPI_D, RPI_E) on proximity subsample
 
Output: 
    figures/regression/
    tables/
"""

import sys
import warnings
warnings.filterwarnings("ignore")
 
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
 
from src.analysis.data_prep import (
    load_merged, add_features, filter_proximity, build_hourly,
)
from config.settings import STORMS, STORM_COLORS, ADVISORY_PREDS
from paths import PATHS
 
try:
    import statsmodels.api as sm
    HAS_SM = True
except ImportError:
    print("Install statsmodels: pip install statsmodels")
    HAS_SM = False

# Paths
FIGURE_DIR = PATHS["figures"] / "regression"
TABLE_DIR  = PATHS["tables"]
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {
    "const":                     "Intercept",
    "delta_forecast_1_wind_z":   "Forecast wind change (z)",
    "delta_wind_kt_z":           "Current wind change (z)",
    "delta_fc_lag1_z":           "Forecast wind change, lag1 (z)",
    "delta_wind_lag1_z":         "Current wind change, lag1 (z)",
    "new_hurricane_warning":     "New hurricane warning†",
    "warning_lag1":              "Warning issued, lag1†",
    "new_hurricane_watch":       "New hurricane watch",
    "storm_category_z":          "Storm category (z)",
    "max_wind_kt_z":             "Current wind level (z)",
    "log_n_posts":               "Log post volume",
    "hour_sin":                  "Hour of day (sin)",
    "hour_cos":                  "Hour of day (cos)",
    "sub_local_community":       "Local community subreddit",
    "sub_state":                 "State subreddit",
    "is_submission":             "Is submission",
}


# OLS with clustered SEs

def run_ols(df, outcome, predictors, cluster_var, model_name):
    df_m = df[[outcome] + predictors + [cluster_var]].dropna()
    print(f"\n{'─' * 50}\nModel: {model_name}  N={len(df_m)}  clusters={df_m[cluster_var].nunique()}")
    if len(df_m) < 20:
        print("  [skip] Insufficient observations")
        return None, None

    X = sm.add_constant(df_m[predictors].astype(float))
    y = df_m[outcome].astype(float)
    result = sm.OLS(y, X).fit(cov_type="cluster",
                               cov_kwds={"groups": df_m[cluster_var]})
    print(result.summary())

    coef_df = pd.DataFrame({
        "variable": result.params.index,
        "coef":     result.params.values,
        "se":       result.bse.values,
        "z":        result.tvalues.values,
        "p_value":  result.pvalues.values,
        "ci_lower": result.conf_int()[0].values,
        "ci_upper": result.conf_int()[1].values,
        "model":    model_name,
        "outcome":  outcome,
        "n_obs":    result.nobs,
        "r2":       result.rsquared,
    })
    coef_df["sig"] = pd.cut(
        coef_df["p_value"],
        bins=[-np.inf, 0.001, 0.01, 0.05, 0.1, np.inf],
        labels=["***", "**", "*", ".", ""],
    )
    return result, coef_df


# Wald test (H3)
def wald_test_h3(result, label=""):
    if result is None:
        return None
    params = result.params.index.tolist()
    if ("delta_forecast_1_wind_z" not in params or
            "delta_wind_kt_z" not in params):
        return None
    idx_fc  = params.index("delta_forecast_1_wind_z")
    idx_cur = params.index("delta_wind_kt_z")
    R = np.zeros((1, len(params)))
    R[0, idx_fc] = 1; R[0, idx_cur] = -1
    wald  = result.wald_test(R, use_f=False)
    chi2  = float(np.asarray(wald.statistic).flat[0])
    p_two = float(np.asarray(wald.pvalue).flat[0])
    b_fc  = result.params["delta_forecast_1_wind_z"]
    b_cur = result.params["delta_wind_kt_z"]
    p_one = p_two / 2 if b_fc > b_cur else 1 - p_two / 2
    print(f"\nWald H3 [{label}]: β_fc={b_fc:+.5f}  β_cur={b_cur:+.5f}  "
          f"p_one={p_one:.4f} "
          f"{'✓' if p_one < 0.05 else '~' if p_one < 0.10 else '✗'}")
    return {"label": label, "b_forecast": round(b_fc, 6),
            "b_current": round(b_cur, 6), "diff": round(b_fc - b_cur, 6),
            "chi2": round(chi2, 4), "p_two": round(p_two, 4),
            "p_one": round(p_one, 4)}


# Incremental R^2
def incremental_r2(df_h, outcome, adv_preds, ctrl_preds, cluster_var, label=""):
    all_preds = ctrl_preds + adv_preds
    df_m = df_h[[outcome] + all_preds + [cluster_var]].dropna()
    if len(df_m) < 20:
        return None

    X_r = sm.add_constant(df_m[ctrl_preds].astype(float))
    y = df_m[outcome].astype(float)
    res_r = sm.OLS(y, X_r).fit(cov_type="cluster",
                                 cov_kwds={"groups": df_m[cluster_var]})
    X_f = sm.add_constant(df_m[all_preds].astype(float))
    res_f = sm.OLS(y, X_f).fit(cov_type="cluster",
                                 cov_kwds={"groups": df_m[cluster_var]})

    r2_c = res_r.rsquared
    r2_f = res_f.rsquared
    dr2 = r2_f - r2_c

    param_names = list(res_f.params.index)
    adv_in_model = [p for p in adv_preds if p in param_names]
    R = np.zeros((len(adv_in_model), len(res_f.params)))
    for i, p in enumerate(adv_in_model):
        R[i, param_names.index(p)] = 1
    wald  = res_f.wald_test(R, use_f=False)
    chi2  = float(np.asarray(wald.statistic).flat[0])
    p_val = float(np.asarray(wald.pvalue).flat[0])

    print(f"  ΔR² [{label} / {outcome}]: "
          f"controls={r2_c:.4f}  full={r2_f:.4f}  "
          f"Δ={dr2:.4f}  joint Chi²={chi2:.3f} p={p_val:.4f}")

    return {"label": label, "outcome": outcome,
            "r2_controls": round(r2_c, 4), "r2_full": round(r2_f, 4),
            "r2_incremental": round(dr2, 4),
            "joint_chi2": round(chi2, 3), "joint_p": round(p_val, 4)}


def _build_hourly_with_subtypes(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    df["is_proximity"] = df["subreddit_type"].isin(["state", "local_community"]).astype(int)
    hourly = build_hourly(df, label=label)
    sub_props = df.groupby(["storm", "hour_floor"]).agg(
        prop_proximity=("is_proximity", "mean"),
    ).reset_index()
    return hourly.merge(sub_props, on=["storm", "hour_floor"], how="left")

# save formatted regression table
def save_table(coef_df, result, stem):
    if coef_df is None:
        return
    main_order = ["const",
                  "delta_forecast_1_wind_z", "delta_wind_kt_z",
                  "delta_fc_lag1_z", "delta_wind_lag1_z",
                  "new_hurricane_warning", "warning_lag1", "new_hurricane_watch",
                  "storm_category_z", "max_wind_kt_z",
                  "log_n_posts", "hour_sin", "hour_cos"]
    storm_order = [c for c in coef_df["variable"] if c.startswith("storm_")]
 
    def fmt(row):
        sig = str(row["sig"]) if str(row["sig"]) != "nan" else ""
        p   = f"{row['p_value']:.3f}" if row["p_value"] >= 0.001 else "<0.001"
        return {"Predictor": LABEL_MAP.get(row["variable"], row["variable"]),
                "β": f"{row['coef']:+.4f}{sig}", "SE": f"({row['se']:.4f})", "p": p}
 
    rows    = [fmt(coef_df[coef_df["variable"] == v].iloc[0])
               for v in main_order if not coef_df[coef_df["variable"] == v].empty]
    fe_rows = [fmt(coef_df[coef_df["variable"] == v].iloc[0])
               for v in storm_order if not coef_df[coef_df["variable"] == v].empty]
    sep     = pd.DataFrame([{"Predictor": "--- Storm FEs ---", "β": "", "SE": "", "p": ""}])
    full    = pd.concat([pd.DataFrame(rows), sep, pd.DataFrame(fe_rows)], ignore_index=True)
    full.to_csv(TABLE_DIR / f"{stem}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"N": int(result.nobs), "R2": round(result.rsquared, 3),
                   "Adj_R2": round(result.rsquared_adj, 3)}]).to_csv(
        TABLE_DIR / f"{stem}_fit.csv", index=False, encoding="utf-8-sig"
    )
    print(f"  Saved: {stem}.csv")

# Main
def run() -> None:   
    print("=" * 60)
    print("Regression Analysis v3")
    print("=" * 60)
    if not HAS_SM:
        return

    df_raw = load_merged()
    df = add_features(df_raw, active_only=True)
    df_prx = filter_proximity(df)
 
    df_h = build_hourly(df, label="full sample")
    df_h_prx = build_hourly(df_prx, label="proximity only")

    def _storm_cols(df_ref):
        return [c for c in df_ref.columns
                if c.startswith("storm_") and
                c not in {"storm_phase", "storm_category", "storm_category_z",
                           "storm_hour_id", "storm_type"}]
 
    def valid(preds, df_ref):
        return [p for p in preds if p in df_ref.columns and df_ref[p].notna().any()]
 
    storm_cols     = _storm_cols(df_h)
    storm_cols_prx = _storm_cols(df_h_prx)
    ctrl_cat       = ["storm_category_z", "log_n_posts", "hour_sin", "hour_cos"] + storm_cols
    ctrl_wind      = ["max_wind_kt_z",    "log_n_posts", "hour_sin", "hour_cos"] + storm_cols
    ctrl_cat_prx   = ["storm_category_z", "log_n_posts", "hour_sin", "hour_cos"] + storm_cols_prx
 
    base_cat      = valid(ADVISORY_PREDS + ctrl_cat,     df_h)
    base_wind     = valid(ADVISORY_PREDS + ctrl_wind,    df_h)
    base_cat_prx  = valid(ADVISORY_PREDS + ctrl_cat_prx, df_h_prx)
 
    wald_rows = []
    incr_rows = []

    # 1. PRIMARY
    print("\n" + "="*60 + "\n1. PRIMARY MODEL — storm_category_z")
    res1, c1 = run_ols(df_h, "mean_rpi_composite", base_cat,
                       "adv_cycle_id", "Primary")
    w = wald_test_h3(res1, "Primary"); wald_rows.append(w) if w else None
    if c1 is not None:
        c1.to_csv(TABLE_DIR/"regression_primary.csv", index=False, encoding="utf-8-sig")
        save_table(c1, res1, "regression_table_primary")
    incr = incremental_r2(df_h,"mean_rpi_composite",ADVISORY_PREDS,ctrl_cat,"adv_cycle_id","Primary")
    if incr: incr_rows.append(incr)

    # 2. ROBUSTNESS A — full composite
    print("\n" + "="*60 + "\n2. ROBUSTNESS A — RPI_composite_full")
    if "mean_rpi_composite_full" in df_h.columns:
        res2, c2 = run_ols(df_h, "mean_rpi_composite_full", base_cat,
                           "adv_cycle_id", "Robustness A")
        w = wald_test_h3(res2, "Robustness A")
        if w: wald_rows.append(w)
        if c2 is not None:
            c2.to_csv(TABLE_DIR / "regression_robustness_full.csv", index=False, encoding="utf-8-sig")
            save_table(c2, res2, "regression_table_robustness_full")
        incr = incremental_r2(df_h, "mean_rpi_composite_full", ADVISORY_PREDS, ctrl_cat,
                               "adv_cycle_id", "Robustness A")
        if incr: incr_rows.append(incr)

    # 3. ROBUSTNESS B — max_wind_kt_z
    print("\n" + "="*60 + "\n3. ROBUSTNESS B — max_wind_kt_z")
    if "max_wind_kt_z" in df_h.columns:
        res3, c3 = run_ols(df_h, "mean_rpi_composite", base_wind,
                           "adv_cycle_id", "Robustness B")
        w = wald_test_h3(res3, "Robustness B")
        if w: wald_rows.append(w)
        if c3 is not None:
            c3.to_csv(TABLE_DIR / "regression_robustness_maxwind.csv", index=False, encoding="utf-8-sig")
            save_table(c3, res3, "regression_table_robustness_maxwind")
        incr = incremental_r2(df_h, "mean_rpi_composite", ADVISORY_PREDS, ctrl_wind,
                               "adv_cycle_id", "Robustness B")
        if incr: incr_rows.append(incr)

    # 4. MULTI-OUTCOME
    print("\n" + "="*60 + "\n4. MULTI-OUTCOME COMPARISON")
    multi_outcomes = [
        ("mean_rpi_composite", "RPI_composite (revised)"),
        ("mean_rpi_composite_full", "RPI_composite (full)"),
        ("mean_rpi_a", "RPI_A (fear)"),
        ("mean_p_likelihood", "p_likelihood (standalone)"),
        ("mean_p_dismissive", "p_dismissive (standalone)"),
    ]
    all_multi_coefs = []
    for oc, ol in multi_outcomes:
        if oc not in df_h.columns:
            continue
        _, cd = run_ols(df_h, oc, valid(base_cat, df_h), "adv_cycle_id", f"Multi: {ol}")
        if cd is not None:
            cd["outcome"] = oc
            all_multi_coefs.append(cd)
        w = wald_test_h3(_, ol)
        if w: wald_rows.append(w)
        incr = incremental_r2(df_h, oc, ADVISORY_PREDS, ctrl_cat, "adv_cycle_id", ol)
        if incr: incr_rows.append(incr)
    if all_multi_coefs:
        pd.concat(all_multi_coefs, ignore_index=True).to_csv(
            TABLE_DIR / "regression_multi_outcome.csv", index=False, encoding="utf-8-sig"
        )

    # 4.5 SUB-SCORE REGRESSIONS — full sample
    print("\n" + "=" * 60 + "\n4.5 SUB-SCORE REGRESSIONS — full sample")
    for oc, ol in [("mean_rpi_a", "RPI_A"), ("mean_p_likelihood", "RPI_D"), ("mean_rpi_e", "RPI_E")]:
        if oc not in df_h.columns:
            continue
        res_sub, cd_sub = run_ols(df_h, oc, valid(base_cat, df_h), "adv_cycle_id", f"Subscore: {ol}")
        if cd_sub is not None:
            save_table(cd_sub, res_sub, f"regression_table_subscore_{oc}_fullsample")
        w = wald_test_h3(res_sub, f"Full {ol}")
        if w: wald_rows.append(w)

    # 5. PROXIMITY (post-level)
    print("\n" + "=" * 60 + "\n5. EXPLORATORY — Community Proximity (post-level)")
    post_sc = [c for c in df.columns
               if c.startswith("storm_") and
               c not in {"storm_phase", "storm_category", "storm_category_z",
                          "storm_hour_id", "storm_type"}]
    post_p  = valid(["delta_forecast_1_wind_z", "delta_wind_kt_z",
                      "storm_category_z", "hour_sin", "hour_cos",
                      "is_submission", "sub_local_community", "sub_state"] + post_sc, df)
    res5, c5 = run_ols(df.dropna(subset=["RPI_composite"] + post_p),
                       "RPI_composite", post_p, "storm_hour_id",
                       "Exploratory Proximity (post-level)")
    if c5 is not None:
        print(c5[c5["variable"].str.startswith("sub_")]
              [["variable", "coef", "se", "p_value", "sig"]].to_string(index=False))
        c5.to_csv(TABLE_DIR / "regression_exploratory_proximity.csv",
                   index=False, encoding="utf-8-sig")
        save_table(c5, res5, "regression_table_exploratory_proximity")

    # 6. PROXIMITY-RESTRICTED SUBSAMPLE
    print("\n" + "="*60 + "\n6. PROXIMITY-RESTRICTED SUBSAMPLE")

    # 6a. Primary model
    res6a, c6a = run_ols(df_h_prx, "mean_rpi_composite", base_cat_prx,
                         "adv_cycle_id", "Proximity — primary")
    w = wald_test_h3(res6a, "Proximity primary")
    if w: wald_rows.append(w)
    if c6a is not None:
        c6a.to_csv(TABLE_DIR / "regression_proximity_primary.csv", index=False, encoding="utf-8-sig")
        save_table(c6a, res6a, "regression_table_proximity_primary")
    incr = incremental_r2(df_h_prx, "mean_rpi_composite", ADVISORY_PREDS,
                           ctrl_cat_prx, "adv_cycle_id", "Proximity primary")
    if incr: incr_rows.append(incr)

    # 6b. Sub-score regressions
    subscore_coefs = []
    for oc, ol in [("mean_rpi_composite", "RPI_composite"),
                   ("mean_rpi_a", "RPI_A"),
                   ("mean_rpi_d", "RPI_D"),
                   ("mean_rpi_e", "RPI_E")]:
        if oc not in df_h_prx.columns:
            continue
        _, cd = run_ols(df_h_prx, oc, base_cat_prx, "adv_cycle_id", f"Proximity: {ol}")
        if cd is not None:
            cd["outcome"] = oc
            subscore_coefs.append((ol, cd))
            cd.to_csv(TABLE_DIR / f"regression_proximity_{oc}.csv",
                       index=False, encoding="utf-8-sig")
        w = wald_test_h3(_, f"Proximity {ol}")
        if w: wald_rows.append(w)
 
    if len(subscore_coefs) > 1:
        pd.concat([cd for _, cd in subscore_coefs]).to_csv(
            TABLE_DIR / "regression_proximity_subscores.csv", index=False, encoding="utf-8-sig"
        )

    # 7 MODERATION ANALYIS (H4)
    print("\n" + "=" * 60 + "\n7. MODERATION ANALYSIS — H4")
    df_h_mod = _build_hourly_with_subtypes(df, label="moderation")

    focal = ["delta_forecast_1_wind_z", "delta_wind_kt_z"]
    for adv in focal:
        if adv in df_h_mod.columns:
            df_h_mod[f"proximity:{adv}"] = df_h_mod["prop_proximity"] * df_h_mod[adv]

    interaction_terms = [c for c in df_h_mod.columns if c.startswith("proximity:")]

    mod_preds = valid(
        ADVISORY_PREDS
        + ["prop_proximity"]
        + interaction_terms
        + ["storm_category_z", "log_n_posts", "hour_sin", "hour_cos"]
        + storm_cols,
        df_h_mod
    )

    mod_coefs = []
    for oc, ol in [("mean_rpi_composite", "RPI_composite"),
                   ("mean_rpi_a",         "RPI_A"),
                   ("mean_p_likelihood",  "RPI_D"),
                   ("mean_rpi_e",         "RPI_E")]:
        if oc not in df_h_mod.columns:
            continue
        _, cd = run_ols(df_h_mod, oc, mod_preds, "adv_cycle_id",
                        f"Moderation H4: {ol}")
        if cd is not None:
            cd["outcome"] = oc
            mod_coefs.append(cd)

    if mod_coefs:
        df_mod = pd.concat(mod_coefs, ignore_index=True)
        df_mod.to_csv(TABLE_DIR / "moderation_interactions.csv",
                      index=False, encoding="utf-8-sig")
        int_summary = df_mod[df_mod["variable"].str.contains(":", na=False)].copy()
        print("\nH4 Interaction term summary:")
        print(int_summary[["model", "variable", "coef", "se", "p_value", "sig"]]
              .to_string(index=False))
        int_summary.to_csv(TABLE_DIR / "moderation_interaction_summary.csv",
                           index=False, encoding="utf-8-sig")
        
    # summary tables
    if wald_rows:
        pd.DataFrame(wald_rows).to_csv(TABLE_DIR / "wald_tests.csv",
                                        index=False, encoding="utf-8-sig")
        print("\nWald tests:\n", pd.DataFrame(wald_rows).to_string(index=False))
    if incr_rows:
        pd.DataFrame(incr_rows).to_csv(TABLE_DIR / "incremental_r2.csv",
                                        index=False, encoding="utf-8-sig")
        print("\nIncremental R²:\n", pd.DataFrame(incr_rows).to_string(index=False))
 
    print(f"\n{'=' * 60}\nDone.\n  Figures: {FIGURE_DIR}\n  Tables: {TABLE_DIR}")

 
if __name__ == "__main__":
    run()