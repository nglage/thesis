"""
Step 5d: Time series visualisations
 
Figures produced:
 
    timeseries_main.png -> 6-panel combined, full sample
    timeseries_rpi_components.png -> 6-panel sub-scores, full sample
    timeseries_full_vs_proximity.png -> full vs proximity overlay
 
    timeseries_main_proximity.png -> 6-panel combined, proximity only
    timeseries_{storm}.png -> individual per-storm plots
"""

import warnings
warnings.filterwarnings("ignore")
 
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
 
from src.analysis.data_prep import (
    load_merged, load_advisories, add_features, filter_proximity, smooth,
)
from config.settings import STORMS, STORM_COLORS, STORM_YEARS, MIN_POSTS_PER_HOUR
from paths import PATHS

# Paths
FIGURE_DIR = PATHS["figures"] / "timeseries"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
 

# Hourly aggregation for timeseries
def make_hourly_ts(df: pd.DataFrame, storm: str) -> pd.DataFrame:
    ds = df[(df["storm"] == storm) & (df["storm_phase"] == "active")].copy()
    ds["hour"] = ds["created_utc"].dt.floor("h")
    agg_cols = [c for c in [
        "RPI_composite", "RPI_A", "RPI_D_rev", "RPI_E_rev",
        "p_likelihood", "p_vulnerable", "fear",
    ] if c in ds.columns]
    agg_dict = {"n_posts": ("RPI_composite", "count")}
    for col in agg_cols:
        agg_dict[col] = (col, "mean")
    hourly = ds.groupby("hour").agg(**agg_dict).reset_index()
    return hourly[hourly["n_posts"] >= MIN_POSTS_PER_HOUR]
 
 
def smooth_col(hourly: pd.DataFrame, col: str, window: int = 6) -> pd.DataFrame:
    return (
        hourly.set_index("hour")[col]
        .rolling(window=window, min_periods=3)
        .mean()
        .reset_index())


# Main timeseries panel
def draw_main_panel(ax, storm: str, hourly: pd.DataFrame, da: pd.DataFrame, color: str):

    # RPI revised — thin raw + bold smoothed
    ax.plot(hourly["hour"], hourly["RPI_composite"],
            color=color, lw=0.6, alpha=0.4, zorder=2)
    s = smooth_col(hourly, "RPI_composite")
    ax.plot(s["hour"], s["RPI_composite"], color=color, lw=2.5, alpha=0.85, zorder=3)
    ax.fill_between(hourly["hour"], hourly["RPI_composite"], alpha=0.04, color=color, zorder=1)

    # Wind axes
    ax2 = ax.twinx()
    ax3 = ax.twinx()
    # Move volume axis just outside wind axis
    ax3.spines["right"].set_position(("axes", 1.06))

    # Current wind — dark dashed
    ax2.step(da["utc_datetime"], da["max_wind_kt"],
             color="#666666", lw=1.4, alpha=0.5, linestyle="--", where="post")
    # Forecast wind — light dotted
    if "forecast_1_wind_kt" in da.columns:
        ax2.step(da["utc_datetime"], da["forecast_1_wind_kt"],
                 color="#222222", lw=1.2, alpha=0.7, linestyle=":", where="post")

    # Post volume — subtle shading on volume axis
    ax3.fill_between(hourly["hour"], hourly["n_posts"],
                     alpha=0.10, color=color, zorder=0)
    ax3.set_ylabel("Posts/h", fontsize=6, color="#999999")
    ax3.tick_params(axis="y", labelsize=5, labelcolor="#999999")
    ax3.set_ylim(0, hourly["n_posts"].max() * 3.5)
    ax3.spines[["top", "left"]].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # Phase boundaries
    for vline, vcolor, text in [
        (da["utc_datetime"].min(), "steelblue", " active →"),
        (da["utc_datetime"].max(), "firebrick",  " post →"),
    ]:
        ax.axvline(vline, color=vcolor, lw=1.2, alpha=0.7, linestyle="--")
        ax.text(vline, 0.63, text, fontsize=7, color=vcolor, va="top")

    # Formatting
    ax.set_ylabel("Mean RPI", fontsize=8, color=color)
    ax2.set_ylabel("Wind speed (kt)", fontsize=8, color="#666666")
    ax.tick_params(axis="y", labelcolor=color, labelsize=7)
    ax2.tick_params(axis="y", labelcolor="#666666", labelsize=7)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(True, alpha=0.2, zorder=0)
    ax.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)
    ax.set_ylim(0.10, 0.68)
    ax2.set_ylim(0, max(da["max_wind_kt"].max() * 1.15, 80))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")
    
    return ax2, ax3


def main_legend():
    return [
        Line2D([0],[0], color="grey", lw=2.5, alpha=0.85, label="RPI composite (6h rolling mean)"),
        Line2D([0],[0], color="#666666", lw=1.4, linestyle="--", label="Current wind speed (kt)"),
        Line2D([0],[0], color="#222222", lw=1.2, linestyle=":", label="12h forecast wind speed (kt)"),
        Line2D([0],[0], color="grey", lw=4, alpha=0.12, label="Post volume (posts/hour)"),
        Line2D([0],[0], color="steelblue", lw=1.2, linestyle="--", label="First advisory (active phase start)"),
        Line2D([0],[0], color="firebrick", lw=1.2, linestyle="--", label="Last advisory (post-storm start)"),
    ]


# Figure 1: Main timeseries combined
def plot_main_combined(df: pd.DataFrame, adv: pd.DataFrame,
                        title_suffix: str = "", suffix: str = "") -> None:
    fig, axes = plt.subplots(len(STORMS), 1, figsize=(14, 3.8 * len(STORMS)), sharex=False)
    title = "Hourly RPI, Current Wind, and 12h Forecast Wind"
    if title_suffix:
        title += f"\n{title_suffix}"
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.002)
 
    for ax, storm in zip(axes, STORMS):
        color = STORM_COLORS[storm]
        da = adv[adv["storm"] == storm].copy()
        hourly = make_hourly_ts(df, storm)
        if hourly.empty:
            ax.set_title(f"{storm} — no data"); continue
        draw_main_panel(ax, storm, hourly, da, color)
        ax.set_title(f"{storm} ({STORM_YEARS[storm]})", fontsize=11, fontweight="bold",
                     color=color, loc="left", pad=4)
 
    fig.legend(handles=main_legend(), loc="lower center", ncol=3, fontsize=8,
               frameon=True, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    out = FIGURE_DIR / f"timeseries_main{suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out.name}")


# Figure 1b: Per-storm individual plots
def plot_main_per_storm(df: pd.DataFrame, adv: pd.DataFrame, suffix: str = "") -> None:
    for storm in STORMS:
        color = STORM_COLORS[storm]
        da = adv[adv["storm"] == storm].copy()
        hourly = make_hourly_ts(df, storm)
        if hourly.empty:
            continue
        fig, ax = plt.subplots(figsize=(12, 4))
        draw_main_panel(ax, storm, hourly, da, color)
        ax.set_title(
            f"Hurricane {storm} ({STORM_YEARS[storm]}) — Hourly RPI, Wind Speed, and 12h Forecast",
            fontsize=11, fontweight="bold",
        )
        ax.legend(handles=main_legend(), fontsize=7, loc="upper left", framealpha=0.8)
        plt.tight_layout()
        out = FIGURE_DIR / f"timeseries_{storm.lower()}{suffix}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out.name}")


# Figure 2: RPI sub-score components
def plot_rpi_components(df, adv, title_suffix="", suffix=""):
    styles = [
        ("RPI_A",     "#E63946", 2.5, "RPI_A: Affective (fear)"),
        ("RPI_D_rev", "#457B9D", 2.5, "RPI_D: Deliberative (p_likelihood)"),
        ("RPI_E_rev", "#2A9D8F", 2.5, "RPI_E: Experiential (p_vulnerable)"),
    ]

    fig, axes = plt.subplots(len(STORMS), 1, figsize=(14, 3.8 * len(STORMS)), sharex=False)
    title = "Hourly RPI Sub-score Trends by Dimension"
    if title_suffix:
        title += f"\n{title_suffix}"
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.002)

    for ax, storm in zip(axes, STORMS):
        color = STORM_COLORS[storm]
        da = adv[adv["storm"] == storm].copy()
        hourly = make_hourly_ts(df, storm)
        if hourly.empty:
            ax.set_title(f"{storm} - no data"); continue

        # Wind on right axis
        ax2 = ax.twinx()
        ax2.step(da["utc_datetime"], da["max_wind_kt"],
                 color="#666666", lw=1.2, alpha=0.5,
                 linestyle="--", where="post",
                 label="Current wind (kt)")
        if "forecast_1_wind_kt" in da.columns:
            ax2.step(da["utc_datetime"], da["forecast_1_wind_kt"],
                     color="#222222", lw=1.0, alpha=0.7,
                     linestyle=":", where="post",
                     label="12h forecast wind (kt)")

        # Sub-score lines
        for col, clr, lw, lbl in styles:
            if col not in hourly.columns:
                continue
            ax.plot(hourly["hour"], hourly[col], color=clr, lw=0.5, alpha=0.18, zorder=2)
            s = smooth_col(hourly, col)
            ax.plot(s["hour"], s[col], color=clr, lw=lw, alpha=0.85, zorder=3, label=lbl)

        # Phase boundaries
        for vline, vcolor in [(da["utc_datetime"].min(), "steelblue"),
                               (da["utc_datetime"].max(), "firebrick")]:
            ax.axvline(vline, color=vcolor, lw=1.0, alpha=0.6, linestyle="--")
 
        ax.set_title(f"{storm} ({STORM_YEARS[storm]})", fontsize=11, fontweight="bold",
                     color=color, loc="left", pad=3)
        ax.set_ylabel("Mean sub-score", fontsize=8)
        ax2.set_ylabel("Wind speed (kt)", fontsize=8, color="#666666")
        ax.set_ylim(0, 1.0)
        ax2.set_ylim(0, da["max_wind_kt"].max() * 1.2)
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax2.tick_params(axis="y", labelsize=6, labelcolor="#666666")
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")
        ax.grid(True, alpha=0.2)
        ax.spines[["top"]].set_visible(False)
        ax2.spines[["top"]].set_visible(False)

        # Combined legend: sub-scores + wind lines
        sub_handles = [Line2D([0],[0], color=c, lw=l, alpha=0.85, label=lbl)
                       for col, c, l, lbl in styles if col in hourly.columns]
        wind_handles = [
            Line2D([0],[0], color="#666666", lw=1.2, linestyle="--", label="Current wind (kt)"),
            Line2D([0],[0], color="#222222", lw=1.0, linestyle=":",  label="12h forecast wind (kt)"),
        ]
        ax.legend(handles=sub_handles + wind_handles, fontsize=7,
                  loc="upper left", framealpha=0.8, ncol=2)
        
    plt.tight_layout()
    out = FIGURE_DIR / f"timeseries_rpi_components{suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out.name}")


# Figure 3: Full vs proximity overlay
def plot_full_vs_proximity(df_full: pd.DataFrame, df_prx: pd.DataFrame,
                            adv: pd.DataFrame) -> None:
    fig, axes = plt.subplots(len(STORMS), 2, figsize=(16, 3.2 * len(STORMS)),
                              sharex=False, sharey=True)
    fig.suptitle(
        "RPI: Full Sample vs Proximity Subsample\n"
        "(state + local community subreddits only)",
        fontsize=12, fontweight="bold", y=1.002,
    )
    for row_idx, storm in enumerate(STORMS):
        color = STORM_COLORS[storm]
        da    = adv[adv["storm"] == storm].copy()
        for col_idx, (df_src, panel_label) in enumerate([(df_full, "Full sample"),
                                                          (df_prx,  "Proximity subsample")]):
            ax     = axes[row_idx, col_idx]
            ax2    = ax.twinx()
            hourly = make_hourly_ts(df_src, storm)
            if hourly.empty:
                ax.set_title(f"{storm} — {panel_label} — no data"); continue
 
            ax.plot(hourly["hour"], hourly["RPI_composite"], color=color, lw=0.5, alpha=0.2)
            s = smooth_col(hourly, "RPI_composite")
            ax.plot(s["hour"], s["RPI_composite"], color=color, lw=2.2, alpha=0.85)
            ax.fill_between(hourly["hour"], hourly["RPI_composite"], alpha=0.04, color=color)

            # Wind lines
            ax2.step(da["utc_datetime"], da["max_wind_kt"],
                     color="#666666", lw=1.2, alpha=0.5,
                     linestyle="--", where="post")
            if "forecast_1_wind_kt" in da.columns:
                ax2.step(da["utc_datetime"], da["forecast_1_wind_kt"],
                         color="#222222", lw=1.0, alpha=0.7,
                         linestyle=":", where="post")

            # Phase boundaries
            for vline, vcolor in [(da["utc_datetime"].min(), "steelblue"),
                                   (da["utc_datetime"].max(), "firebrick")]:
                ax.axvline(vline, color=vcolor, lw=1.0, alpha=0.6, linestyle="--")
 
            mean_rpi = hourly["RPI_composite"].mean()
            ax.set_title(f"{storm} - {panel_label} (μ={mean_rpi:.3f})",
                         fontsize=9, fontweight="bold", color=color)
            ax.set_ylabel("Mean RPI" if col_idx == 0 else "", fontsize=8, color=color)
            ax2.set_ylabel("Wind speed (kt)" if col_idx == 1 else "", fontsize=7, color="#666666")
            ax.set_ylim(0.10, 0.70)
            ax2.set_ylim(0, da["max_wind_kt"].max() * 1.2)
            ax.tick_params(axis="x", labelsize=7)
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")
            ax.tick_params(axis="y", labelsize=7, labelcolor=color)
            ax2.tick_params(axis="y", labelsize=6, labelcolor="#666666")
            ax.grid(True, alpha=0.2)
            ax.spines[["top"]].set_visible(False)
            ax2.spines[["top"]].set_visible(False)

    # Shared legend at bottom
    legend_handles = [
        Line2D([0],[0], color="grey",      lw=2.2, alpha=0.85, label="RPI composite (6h rolling mean)"),
        Line2D([0],[0], color="#666666",   lw=1.2, linestyle="--", label="Current wind speed (kt)"),
        Line2D([0],[0], color="#222222",   lw=1.0, linestyle=":",  label="12h forecast wind speed (kt)"),
        Line2D([0],[0], color="steelblue", lw=1.2, linestyle="--", label="First advisory"),
        Line2D([0],[0], color="firebrick", lw=1.2, linestyle="--", label="Last advisory"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=8,
               frameon=True, bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout()
    out = FIGURE_DIR / "timeseries_full_vs_proximity.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out.name}")


# Main
def run() -> None:
    print("=" * 60)
    print("Time Series Plots")
    print("=" * 60)
 
    df_raw = load_merged()
    adv = load_advisories()
    df = add_features(df_raw, active_only=False)
    df_prx = filter_proximity(df)
 
    df_active = df[df["storm_phase"] == "active"].copy()
    df_prx_active = df_prx[df_prx["storm_phase"] == "active"].copy()
 
    print("\nFigure 1: Main timeseries (full sample)...")
    plot_main_combined(df_active, adv)
 
    print("\nFigure 2: Sub-score components (full sample)...")
    plot_rpi_components(df_active, adv)
 
    print("\nFigure 3: Full vs proximity overlay...")
    plot_full_vs_proximity(df_active, df_prx_active, adv)
 
    print("\nAppendix: Per-storm plots (full sample)...")
    plot_main_per_storm(df_active, adv)
 
    print("\nAppendix: Main timeseries (proximity subsample)...")
    plot_main_combined(
        df_prx_active, adv,
        title_suffix="State + local community subreddits only",
        suffix="_proximity",
    )
 
    print(f"\nDone. Figures saved to: {FIGURE_DIR}")
 
 
if __name__ == "__main__":
    run()
 