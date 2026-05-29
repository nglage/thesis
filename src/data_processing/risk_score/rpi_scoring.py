"""
- Step 3: Score Reddit posts with the Risk Perception Index (RPI)
 
For each storm:
  1. Load filtered submissions + comments from Step 2b.
  2. Apply NLI relevance filter (facebook/bart-large-mnli).
  3. Score retained posts with:
       - Emotion classifier (j-hartmann/emotion-english-distilroberta-base) → RPI_A
       - Zero-shot NLI (bart-large-mnli) for discourse labels → RPI_D, RPI_E
       - VADER sentiment (baseline / validity check)
  4. Compute variance-weighted composite RPI.
  5. Save per-storm and pooled scored parquet files, plus weights JSON.
 
Output: data/processed/reddit/scored/{Storm}_scored.parquet
        data/processed/reddit/scored/all_storms_scored.parquet
        data/processed/reddit/scored/rpi_weights.json
"""

import json
import warnings
warnings.filterwarnings("ignore")
 
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm
 
from config.settings import (
    STORMS,
    NLI_MODEL, EMOTION_MODEL,
    LABEL_FEAR, LABEL_DISMISSIVE, LABEL_LIKELIHOOD,
    LABEL_UNCERTAINTY, LABEL_VULNERABLE, LABEL_PROTECTIVE, LABEL_FACTUAL,
    DISCOURSE_LABELS,
    NLI_RELEVANCE_THRESHOLD, SCORING_BATCH_SIZE, MAX_TEXT_CHARS, LOG_EPSILON,
)
from paths import PATHS

# Paths
INPUT_DIR  = PATHS["data_processed"] / "reddit" / "filtered"
OUTPUT_DIR = PATHS["data_processed"] / "reddit" / "scored"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Device
DEVICE      = 0 if torch.cuda.is_available() else -1
DEVICE_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"


def load_models():
    print(f"\nRunning on: {DEVICE_NAME}")
    print("Loading models...")
    nli_pipe = pipeline(
        "zero-shot-classification",
        model=NLI_MODEL,
        device=DEVICE,
    )
    emotion_pipe = pipeline(
        "text-classification",
        model=EMOTION_MODEL,
        top_k=None,
        device=DEVICE,
        truncation=True,
        max_length=512,
    )
    print("Models loaded.\n")
    return nli_pipe, emotion_pipe

def batch_nli(nli_pipe, texts, labels, multi_label=False):
    results = []
    for i in tqdm(range(0, len(texts), SCORING_BATCH_SIZE), desc="  NLI", leave=False):
        batch = texts[i: i + SCORING_BATCH_SIZE]
        out = nli_pipe(batch, candidate_labels=labels, multi_label=multi_label)
        if isinstance(out, dict):
            out = [out]
        for item in out:
            results.append(dict(zip(item["labels"], item["scores"])))
    return results


def batch_emotion(emotion_pipe, texts):
    results = []
    for i in tqdm(range(0, len(texts), SCORING_BATCH_SIZE), desc="  Emotion", leave=False):
        batch = texts[i: i + SCORING_BATCH_SIZE]
        out = emotion_pipe(batch)
        if isinstance(out[0], dict):
            out = [out]
        for item in out:
            results.append({d["label"]: d["score"] for d in item})
    return results

# per-storm scoring
def score_storm(storm: str, nli_pipe, emotion_pipe) -> pd.DataFrame | None:
    print(f"\n{'='*60}\nPass 1 — {storm}")
    dfs = []
    for path, label in [
        (INPUT_DIR / f"{storm}_submissions_filtered.parquet", "submissions"),
        (INPUT_DIR / f"{storm}_comments_filtered.parquet",    "comments"),
    ]:
        if path.exists():
            d = pd.read_parquet(path)
            d["text_for_scoring"] = d.get(
                "text_clean",
                d.get("body_clean", pd.Series(dtype=str)),
            ).fillna("")
            dfs.append(d)
            print(f"  {label}: {len(d)}")
        else:
            print(f"  [skip] {path.name}")
 
    if not dfs:
        return None
    
    df = pd.concat(dfs, ignore_index=True)
    df["created_utc"]      = pd.to_datetime(df["created_utc"], utc=True)
    df["text_for_scoring"] = df["text_for_scoring"].str[:MAX_TEXT_CHARS]
    texts = df["text_for_scoring"].tolist()

    # Relevance filter
    print(f"\n  Relevance filter (threshold={NLI_RELEVANCE_THRESHOLD})")
    rel = batch_nli(
        nli_pipe,
        texts,
        ["directly discussing a hurricane or its impacts", "unrelated to a hurricane"],
        multi_label=False,
    )
    df["p_hurricane_relevant"] = [
        s["directly discussing a hurricane or its impacts"] for s in rel
    ]
    df = df[df["p_hurricane_relevant"] >= NLI_RELEVANCE_THRESHOLD].copy()
    print(f"  Retained: {len(df)}")
    if df.empty:
        return None
 
    texts_r = df["text_for_scoring"].tolist()

    # Emotion classification
    print("  Emotion classification")
    emo = batch_emotion(emotion_pipe, texts_r)
    for e in ["fear", "sadness", "surprise", "anger", "joy", "disgust", "neutral"]:
        df[e] = [s.get(e, 0.0) for s in emo]

    print("  Discourse classification")
    disc = batch_nli(nli_pipe, texts_r, DISCOURSE_LABELS, multi_label=True)
    df["p_fear_discourse"] = [s.get(LABEL_FEAR, 0.0) for s in disc]
    df["p_dismissive"] = [s.get(LABEL_DISMISSIVE, 0.0) for s in disc]
    df["p_likelihood"] = [s.get(LABEL_LIKELIHOOD, 0.0) for s in disc]
    df["p_uncertainty"] = [s.get(LABEL_UNCERTAINTY, 0.0) for s in disc]
    df["p_vulnerable"] = [s.get(LABEL_VULNERABLE, 0.0) for s in disc]
    df["p_protective"] = [s.get(LABEL_PROTECTIVE, 0.0) for s in disc]
    df["p_factual"] = [s.get(LABEL_FACTUAL, 0.0) for s in disc]

    # PRI sub-scores
    df["RPI_A"] = df["fear"]
    df["RPI_A_log"] = np.log(df["fear"] + LOG_EPSILON)
    df["RPI_D_full"] = df["p_uncertainty"] * 0.5 + df["p_likelihood"] * 0.5
    df["RPI_E_full"] = df["p_vulnerable"]  * 0.5 + df["p_protective"] * 0.5
    df["RPI_D_rev"] = df["p_likelihood"]
    df["RPI_E_rev"] = df["p_vulnerable"]
    return df

# variance-based composite weights
def estimate_weights(all_dfs: list[pd.DataFrame]) -> dict:
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\n{'=' * 60}\nVariance weights (pooled):")
    weights = {}
    for spec, d_col, e_col in [
        ("revised", "RPI_D_rev",  "RPI_E_rev"),
        ("full",    "RPI_D_full", "RPI_E_full"),
    ]:
        sa = combined["RPI_A"].std()
        sd = combined[d_col].std()
        se = combined[e_col].std()
        t  = sa + sd + se + 1e-8
        wa, wd, we = sa / t, sd / t, se / t
        print(f"  {spec}: w_A={wa:.4f}  w_D={wd:.4f}  w_E={we:.4f}")
        weights[spec] = {
            "w_A": round(wa, 6), "w_D": round(wd, 6), "w_E": round(we, 6),
            "std_A": round(sa, 6), "std_D": round(sd, 6), "std_E": round(se, 6),
        }
    return weights

def apply_composite(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    wr = weights["revised"]
    df["RPI_composite"] = (
        wr["w_A"] * df["RPI_A"] + wr["w_D"] * df["RPI_D_rev"] + wr["w_E"] * df["RPI_E_rev"]
    ).clip(0, 1)
 
    wf = weights["full"]
    df["RPI_composite_full"] = (
        wf["w_A"] * df["RPI_A"] + wf["w_D"] * df["RPI_D_full"] + wf["w_E"] * df["RPI_E_full"]
    ).clip(0, 1)
 
    df["rpi_w_A"] = wr["w_A"]
    df["rpi_w_D"] = wr["w_D"]
    df["rpi_w_E"] = wr["w_E"]
    return df
 
 
def save_scored(df: pd.DataFrame, storm: str) -> None:
    save_cols = [
        "id", "author", "created_utc", "subreddit", "storm", "type", "score",
        "text_for_scoring", "p_hurricane_relevant",
        "fear", "sadness", "surprise", "anger", "joy", "disgust", "neutral",
        "p_fear_discourse", "p_dismissive", "p_likelihood", "p_uncertainty",
        "p_vulnerable", "p_protective", "p_factual",
        "RPI_simple", "RPI_A", "RPI_A_log",
        "RPI_D_rev", "RPI_E_rev", "RPI_composite",
        "RPI_D_full", "RPI_E_full", "RPI_composite_full",
        "rpi_w_A", "rpi_w_D", "rpi_w_E",
    ]
    save_cols = [c for c in save_cols if c in df.columns]
    df[save_cols].to_parquet(OUTPUT_DIR / f"{storm}_scored.parquet", index=False)
    print(
        f"  {storm}: {len(df)} rows  "
        f"RPI_composite μ={df['RPI_composite'].mean():.4f} "
        f"σ={df['RPI_composite'].std():.4f}"
    )

# Main

def run() -> None:
    print("=" * 60)
    print("RPI Scoring Pipeline")
    print("Spec: RPI_D=p_likelihood, RPI_E=p_vulnerable, variance-weighted")
    print("=" * 60)
 
    nli_pipe, emotion_pipe = load_models()
 
    all_dfs = []
    for storm in STORMS:
        df = score_storm(storm, nli_pipe, emotion_pipe)
        if df is not None:
            df["storm"] = storm
            all_dfs.append(df)
 
    if not all_dfs:
        print("No data. Exiting.")
        return
 
    weights = estimate_weights(all_dfs)
 
    weights_path = OUTPUT_DIR / "rpi_weights.json"
    with open(weights_path, "w") as f:
        json.dump(
            {
                **weights,
                "log_epsilon": LOG_EPSILON,
                "note": (
                    "RPI_D=p_likelihood, RPI_E=p_vulnerable, "
                    "variance-weighted no floor. "
                    "Full composite uses 4-label combination (robustness only)."
                ),
            },
            f,
            indent=2,
        )
    print(f"\nWeights saved to {weights_path}")
 
    print("\nApplying composite and saving per-storm files...")
    for df in all_dfs:
        df = apply_composite(df, weights)
        save_scored(df, df["storm"].iloc[0])
 
    # Pooled file
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all = apply_composite(df_all, weights)
    df_all.to_parquet(OUTPUT_DIR / "all_storms_scored.parquet", index=False)
 
    print(f"\nDone. Saved to: {OUTPUT_DIR}")
 
 
if __name__ == "__main__":
    run()
