"""
Step 5a: Descriptive statistics and measurement validation.
 
Produces:
  - Corpus overview table
  - RPI summary statistics by storm
  - Discourse label distributions
  - Construct, discriminant, and convergent validity checks
  - Volume-RPI correlation
  - Example post rows
  - Figures: distributions, validity scatter plots, discourse bar chart, RPI by phase, RPI by subreddit type, post volume timeseries
 
Output: 
    figures/descriptives/
    tables/descriptives/
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
 
from config.settings import STORMS, STORM_COLORS, SUBREDDIT_TYPE_MAP, MIN_POSTS_PER_HOUR
from src.analysis.data_prep import load_merged, load_advisories
from paths import PATHS

# Paths
FIGURE_DIR = PATHS["figures"] / "descriptives"
TABLE_DIR  = PATHS["tables"]  / "descriptives"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True,  exist_ok=True)
 
SUBREDDIT_TYPE_LABELS = {
    "general_weather": "General weather",
    "state":  "State",
    "local_community": "Local community",
}
PHASE_ORDER  = ["pre_advisory", "active", "post_storm"]
PHASE_LABELS = {
    "pre_advisory": "Pre-advisory",
    "active":       "Active storm",
    "post_storm":   "Post-storm",
}

# 1. Corpus overview
def describe_corpus(df: pd.DataFrame, adv: pd.DataFrame) -> pd.DataFrame:
    print("\n[1] Corpus overview")

    adv_counts = adv.groupby("storm").size()

    rows = []
    for storm in STORMS:
        ds = df[df["storm"] == storm]
        active = ds[ds["storm_phase"] == "active"]
        active_clean = active[active["author"] != "[deleted]"]
        rows.append({
            "storm": storm,
            "n_total": len(ds),
            "n_submissions": (ds["type"] == "submission").sum(),
            "n_comments": (ds["type"] == "comment").sum(),
            "n_active": len(active),
            "n_pre_advisory": (ds["storm_phase"] == "pre_advisory").sum(),
            "n_post_storm": (ds["storm_phase"] == "post_storm").sum(),
            "n_general_weather": (ds["subreddit_type"] == "General weather").sum(),
            "n_state": (ds["subreddit_type"] == "State").sum(),
            "n_local_community": (ds["subreddit_type"] == "Local community").sum(),
            "date_first": ds["created_utc"].min().strftime("%Y-%m-%d"),
            "date_last": ds["created_utc"].max().strftime("%Y-%m-%d"),
            "n_subreddits": ds["subreddit"].nunique(),
            "n_advisories": adv_counts.get(storm, 0),
            "n_unique_users": active_clean["author"].nunique(),
        })
    corpus_df = pd.DataFrame(rows)
    totals = corpus_df[[c for c in corpus_df.columns if c.startswith("n_")]].sum()
    totals["storm"] = "TOTAL"

    active_all = df[df["storm_phase"] == "active"]
    active_all_clean = active_all[active_all["author"] != "[deleted]"]
    totals["n_unique_users"] = active_all_clean["author"].nunique()

    corpus_df = pd.concat([corpus_df, pd.DataFrame([totals])], ignore_index=True)
    print(corpus_df.to_string(index=False))
    corpus_df.to_csv(TABLE_DIR / "corpus.csv", index=False, encoding="utf-8-sig")
    return corpus_df

# 1a Description of sparse hours
def describe_sparse_hours(df: pd.DataFrame) -> None:
    print("\n[7] Sparse hours report")
    df = df[df["storm_phase"] == "active"].copy()
    df["hour_floor"] = df["created_utc"].dt.floor("h")

    hourly_all = (
        df.groupby(["storm", "hour_floor"])
        .agg(n_posts=("RPI_composite", "count"))
        .reset_index()
    )
    hourly_all["dropped"] = hourly_all["n_posts"] < MIN_POSTS_PER_HOUR

    total   = len(hourly_all)
    dropped = hourly_all["dropped"].sum()
    print(f"  Total active storm-hours: {total}")
    print(f"  Dropped (<{MIN_POSTS_PER_HOUR} posts): {dropped} ({100*dropped/total:.1f}%)")
    print(f"  Kept:                     {total-dropped} ({100*(total-dropped)/total:.1f}%)")

    summary = (
        hourly_all.groupby("storm")
        .agg(
            total_hours  =("n_posts", "count"),
            dropped_hours=("dropped",  "sum"),
            median_posts =("n_posts",  "median"),
        )
        .assign(pct_dropped=lambda x: (x["dropped_hours"] / x["total_hours"] * 100).round(1))
        .reindex(STORMS)
    )
    print(f"\n  By storm:")
    print(summary.to_string())
    summary.to_csv(TABLE_DIR / "sparse_hours_report.csv", encoding="utf-8-sig")
    print(f"  Saved: sparse_hours_report.csv")


# 2. RPI summary
def describe_rpi(df):
    print("\n[2] RPI summary (active phase)")
    active = df[df["storm_phase"] == "active"]
    rows = []
    for storm in STORMS:
        ds = active[active["storm"] == storm]
        for col in ["RPI_simple", "RPI_composite", "RPI_composite_full",
                    "RPI_A", "RPI_D_rev", "RPI_E_rev"]:
            if col not in ds.columns:
                continue
            rows.append({
                "storm": storm, "measure": col, "n": len(ds),
                "mean":   ds[col].mean().round(4),
                "median": ds[col].median().round(4),
                "sd":     ds[col].std().round(4),
                "p25":    ds[col].quantile(0.25).round(4),
                "p75":    ds[col].quantile(0.75).round(4),
            })
    rpi_df = pd.DataFrame(rows)
    print(rpi_df[rpi_df["measure"].isin(["RPI_composite", "RPI_composite_full"])
                 ].to_string(index=False))
    rpi_df.to_csv(TABLE_DIR / "rpi_summary.csv", index=False, encoding="utf-8-sig")
    return rpi_df


# 3. Discourse label summary
def describe_discourse(df):
    print("\n[3] Discourse label summary (active phase)")
    active = df[df["storm_phase"] == "active"]
    disc_cols = ["p_dismissive", "p_protective", "p_uncertainty",
                 "p_likelihood", "p_vulnerable", "p_factual"]
    disc_cols = [c for c in disc_cols if c in active.columns]
    rows = []
    for storm in STORMS:
        ds = active[active["storm"] == storm]
        row = {"storm": storm, "n": len(ds)}
        for col in disc_cols:
            row[f"mean_{col}"] = ds[col].mean().round(3)
            row[f"std_{col}"]  = ds[col].std().round(3)
        rows.append(row)
    disc_df = pd.DataFrame(rows)
    print(disc_df[["storm"] + [f"mean_{c}" for c in disc_cols]].to_string(index=False))
    disc_df.to_csv(TABLE_DIR / "discourse_summary.csv", index=False, encoding="utf-8-sig")
    return disc_df

# 4. Construct + convergent validity
def describe_validity(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[4] Validity correlations")
    active = df[df["storm_phase"] == "active"].copy()

    corr_cols = ["RPI_simple", "RPI_composite", "RPI_composite_full",
                 "RPI_A", "RPI_D_rev", "RPI_E_rev"]
    corr_cols = [c for c in corr_cols if c in active.columns]
    corr_mat = active[corr_cols].corr().round(3)
    print("\nConstruct validity:")
    print(corr_mat)
    corr_mat.to_csv(TABLE_DIR / "validity_corr.csv", encoding="utf-8-sig")

    # Convergent validity: p_fear_discourse vs fear
    if "p_fear_discourse" in active.columns and "fear" in active.columns:
        print("\nConvergent validity: p_fear_discourse vs fear (emotion classifier)")
        conv_rows = []
        for storm in ["Overall"] + STORMS:
            ds = active if storm == "Overall" else active[active["storm"] == storm]
            ds = ds[["p_fear_discourse", "fear"]].dropna()
            if len(ds) < 10:
                continue
            r, p = stats.pearsonr(ds["p_fear_discourse"], ds["fear"])
            print(f"  {storm}: r={r:.3f}, p={p:.4f}")
            conv_rows.append({"storm": storm, "r": round(r, 3), "p": round(p, 4)})
        pd.DataFrame(conv_rows).to_csv(
            TABLE_DIR / "convergent_validity.csv",
            index=False, encoding="utf-8-sig")

    return corr_mat


# 5. Volume-RPI correlation
def describe_volume_rpi(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[5] Volume-RPI correlation (hourly)")
    active = df[df["storm_phase"] == "active"].copy()
    active["hour"] = active["created_utc"].dt.floor("h")
    rows   = []
    for storm in STORMS + ["Pooled"]:
        ds = active if storm == "Pooled" else active[active["storm"] == storm]
        by = ["storm", "hour"] if storm == "Pooled" else ["hour"]
        hourly = ds.groupby(by).agg(
            mean_rpi=("RPI_composite", "mean"),
            n_posts =("RPI_composite", "count"),
        ).reset_index()
        hourly = hourly[hourly["n_posts"] >= 5]
        if len(hourly) < 5:
            continue
        r, p = stats.pearsonr(hourly["n_posts"], hourly["mean_rpi"])
        print(f"  {storm}: r={r:.3f}, p={p:.4f}")
        rows.append({"storm": storm, "r": round(r, 3), "p": round(p, 4), "n_hours": len(hourly)})
    vol_df = pd.DataFrame(rows)
    vol_df.to_csv(TABLE_DIR / "volume_rpi_corr.csv", index=False, encoding="utf-8-sig")
    return vol_df


# 6. Example rows
def output_example_rows(df: pd.DataFrame) -> None:
    print("\n[6] Example rows")
    active = df[
        (df["storm_phase"] == "active") &
        df["advisory_utc"].notna()
    ].copy()

    display_cols = [
        "storm", "created_utc", "subreddit", "subreddit_type", "type",
        "text_for_scoring", "RPI_simple", "RPI_composite",
        "p_fear_discourse", "fear", "p_uncertainty", "p_likelihood",
        "p_vulnerable", "p_protective", "p_dismissive",
        "advisory_utc", "max_wind_kt", "forecast_1_wind_kt",
        "delta_forecast_1_wind", "storm_category", "new_hurricane_warning",
    ]
    display_cols = [c for c in display_cols if c in active.columns]

    examples = pd.concat([
        active.nlargest(50, "RPI_composite").sample(1, random_state=42),
        active.nlargest(50, "p_dismissive").sample(1, random_state=42),
        active.nlargest(50, "p_vulnerable").sample(1, random_state=42),
        active.nlargest(50, "p_likelihood").sample(1, random_state=42),
    ])[display_cols]
    examples["label"] = ["High RPI", "High dismissive", "High vulnerable", "High likelihood"]
    examples.to_csv(TABLE_DIR / "example_rows.csv", index=False, encoding="utf-8-sig")
    print("  Example rows saved.")


# Plots
def plot_rpi_distributions(df: pd.DataFrame) -> None:
    active = df[df["storm_phase"] == "active"]
    specs  = [
        ("RPI_composite", "Composite RPI"),
        ("RPI_A",         "Affective (RPI_A)"),
        ("RPI_D_rev",     "Deliberative (RPI_D)"),
        ("RPI_E_rev",     "Experiential (RPI_E)"),
    ]
    fig, axes = plt.subplots(4, 6, figsize=(18, 12), sharey=False)
    fig.suptitle("RPI Score Distributions by Storm", fontsize=13, fontweight="bold")
    for col_idx, storm in enumerate(STORMS):
        ds    = active[active["storm"] == storm]
        color = STORM_COLORS[storm]
        for row_idx, (col, label) in enumerate(specs):
            if col not in ds.columns:
                continue
            ax   = axes[row_idx, col_idx]
            vals = ds[col].dropna()
            ax.hist(vals, bins=40, color=color, alpha=0.7, edgecolor="none")
            ax.axvline(vals.mean(), color="black", lw=1.5, linestyle="--",
                       label=f"μ={vals.mean():.3f}")
            ax.set_title(storm if row_idx == 0 else "", fontsize=10,
                         fontweight="bold", color=color)
            ax.set_xlabel(label, fontsize=8)
            ax.set_ylabel("Count" if col_idx == 0 else "", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7)
            ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "rpi_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: rpi_distributions.png")


def plot_validity(df):
    active = df[df["storm_phase"] == "active"].copy()
    has_conv = ("p_fear_discourse" in active.columns and
                     "fear" in active.columns)
    n_panels = 2 if has_conv else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    fig.suptitle("Construct & Convergent Validity",
                 fontsize=12, fontweight="bold")
    
    pairs = [
        ("RPI_A", "RPI_composite", "RPI_A vs RPI_composite\n(construct validity)",
         "RPI_A (fear)", "RPI_composite (revised)"),
    ]
    if has_conv:
        pairs.append(
            ("fear", "p_fear_discourse", "Emotion classifier vs NLI fear\n(convergent validity)",
             "fear (emotion classifier)", "p_fear_discourse (NLI)")
        )
    
    # RPI_simple vs RPI_composite
    for ax, (x_col, y_col, title, xlabel, ylabel) in zip(axes, pairs): 
        for storm in STORMS:
            ds = active[active["storm"] == storm].sample(
                min(500, len(active[active["storm"] == storm])), random_state=42)
            ax.scatter(ds[x_col], ds[y_col],
                    color=STORM_COLORS[storm], alpha=0.15, s=8, label=storm)
        valid = active[[x_col, y_col]].dropna()
        r, _ = stats.pearsonr(valid[x_col], valid[y_col])
        m, b = np.polyfit(valid[x_col], valid[y_col], 1)
        xl = np.linspace(valid[x_col].min(), valid[x_col].max(), 100)
        ax.plot(xl, m * xl + b, color="black", lw=1.5, label=f"r={r:.3f}")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, markerscale=2)
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "validity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: validity.png")


def plot_discourse_by_storm(df: pd.DataFrame) -> None:
    active = df[df["storm_phase"] == "active"].copy()
    disc_cols = {k: v for k, v in {
        "p_dismissive": "Dismissive", "p_protective": "Protective",
        "p_uncertainty": "Uncertainty", "p_likelihood": "Likelihood",
        "p_vulnerable": "Vulnerable", "p_factual": "Factual",
    }.items() if k in active.columns}
    means = (
        active.groupby("storm")[list(disc_cols.keys())]
        .mean().reindex(STORMS).rename(columns=disc_cols)
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(STORMS))
    n = len(means.columns)
    w = 0.8 / n
    cmap = plt.cm.Set2(np.linspace(0, 1, n))
    for i, col in enumerate(means.columns):
        ax.bar(x + (i - n / 2 + 0.5) * w, means[col], w,
               color=cmap[i], alpha=0.8, label=col)
    ax.set_xticks(x)
    ax.set_xticklabels(STORMS, fontsize=10)
    ax.set_ylabel("Mean probability score", fontsize=10)
    ax.set_title("Mean Discourse Label Scores by Storm (active phase)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "discourse_by_storm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: discourse_by_storm.png")


def plot_rpi_by_phase(df: pd.DataFrame) -> None:
    fig, ax  = plt.subplots(figsize=(10, 5))
    x        = np.arange(len(STORMS))
    width    = 0.25
    phases   = [p for p in PHASE_ORDER if p in df["storm_phase"].unique()]
    colors   = {"pre_advisory": "#90BE6D", "active": "#F4A261", "post_storm": "#C1121F"}
    for i, phase in enumerate(phases):
        means = [
            df[(df["storm"] == s) & (df["storm_phase"] == phase)]["RPI_composite"].mean()
            for s in STORMS
        ]
        sems = [
            df[(df["storm"] == s) & (df["storm_phase"] == phase)]["RPI_composite"].sem()
            for s in STORMS
        ]
        offset = (i - len(phases) / 2 + 0.5) * width
        ax.bar(x + offset, means, width, color=colors.get(phase, "#999"),
               alpha=0.7, label=PHASE_LABELS.get(phase, phase))
        ax.errorbar(x + offset, means, yerr=sems,
                    fmt="none", color="black", capsize=3, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(STORMS, fontsize=10)
    ax.set_ylabel("Mean RPI_composite", fontsize=10)
    ax.set_title("Mean RPI_composite by Storm Phase", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "rpi_by_phase.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: rpi_by_phase.png")


def plot_rpi_by_subreddit(df):
    active = df[df["storm_phase"] == "active"].copy()
    sub_order  = ["General weather", "State", "Local community"]
    sub_colors = {"General weather": "#6B9CC7",
                  "State":  "#F4A261",
                  "Local community": "#E63946"}
    fig, axes = plt.subplots(1, 6, figsize=(18, 5), sharey=True)
    fig.suptitle("RPI_composite by Subreddit Type and Storm (active phase)",
                 fontsize=12, fontweight="bold")
    for ax, storm in zip(axes, STORMS):
        ds = active[active["storm"] == storm]
        data   = [ds[ds["subreddit_type"]==s]["RPI_composite"].dropna().values
                  for s in sub_order]
        labels = [s for s, d in zip(sub_order, data) if len(d) > 0]
        data   = [d for d in data if len(d) > 0]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                        medianprops={"color":"black","lw":1.5})
        for patch, label in zip(bp["boxes"], labels):
            patch.set_facecolor(sub_colors.get(label, "#999"))
            patch.set_alpha(0.7)
        ax.set_title(storm, fontsize=10, fontweight="bold")
        ax.set_xticklabels([l.replace(" ","\n") for l in labels], fontsize=6)
        ax.set_ylabel("RPI_composite" if storm == "Harvey" else "", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    out = FIGURE_DIR / "rpi_by_subreddit.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def plot_volume_timeseries(df: pd.DataFrame) -> None:
    sub_colors = {
        "General weather": "#6B9CC7",
        "State":  "#F4A261",
        "Local community": "#E63946",
    }
    fig, axes = plt.subplots(len(STORMS), 1, figsize=(14, 3 * len(STORMS)), sharex=False)
    fig.suptitle("Post Volume Over Time by Subreddit Type", fontsize=12, fontweight="bold", y=1.002)
    for ax, storm in zip(axes, STORMS):
        ds = df[df["storm"] == storm].copy()
        ds["hour"] = ds["created_utc"].dt.floor("h")
        for sub_type, color in sub_colors.items():
            subset = ds[ds["subreddit_type"] == sub_type]
            if subset.empty:
                continue
            hourly = subset.groupby("hour").size().reset_index(name="n")
            ax.fill_between(hourly["hour"], hourly["n"], alpha=0.5, color=color, label=sub_type)
            ax.plot(hourly["hour"], hourly["n"], color=color, lw=0.8, alpha=0.7)
        ax.set_title(storm, fontsize=10, fontweight="bold", loc="left")
        ax.set_ylabel("Posts/hour", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=6, loc="upper right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "volume_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: volume_timeseries.png")

# Main
def run() -> None:
    print("=" * 60)
    print("Descriptive Analysis")
    print("=" * 60)
 
    df, adv = load_merged(), load_advisories()
    _type_display = {
        "general_weather": "General weather",
        "state": "State",
        "local_community": "Local community",
    }
    df["subreddit_type"] = (
        df["subreddit"]
        .map(SUBREDDIT_TYPE_MAP)
        .map(_type_display)
        .fillna("General weather")
    )
 
    print(f"Total posts: {len(df)}")
 
    describe_corpus(df, adv)
    describe_sparse_hours(df)
    describe_rpi(df)
    describe_discourse(df)
    describe_validity(df)
    describe_volume_rpi(df)
    output_example_rows(df)
 
    print("\nGenerating figures...")
    plot_rpi_distributions(df)
    plot_validity(df)
    plot_discourse_by_storm(df)
    plot_rpi_by_phase(df)
    plot_volume_timeseries(df)

    print(f"\nDone.\n  Tables:  {TABLE_DIR}\n  Figures: {FIGURE_DIR}")
 
 
if __name__ == "__main__":
    run()