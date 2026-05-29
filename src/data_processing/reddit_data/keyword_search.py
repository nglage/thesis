"""
- Part A: Sample general subreddits for manual inspection
- Part B: TF-IDF + KeyBERT keyword extraction from verified hurricane content
- Run part A first, inspect csv, then run part B

- script does not need to be re-run unless corpus or subreddit selection changes
"""

import pandas as pd
from pathlib import Path

from paths import PATHS
from config.settings import STORMS, GENERAL_SUBREDDITS

INPUT_DIR  = PATHS["data_processed"] / "reddit" / "time_filtered"
OUTPUT_DIR = PATHS["data_processed"] / "reddit" / "keyword_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Part A: Sample general subreddits for inspection

def sample_general_subreddits(n_per_storm: int = 50):
    """
    Sample n posts per storm from general subreddits only,
    export as CSV for manual inspection before using as ground truth
    """
    print("Sampling general subreddits for inspection...\n")

    all_subs = []
    all_coms = []

    for storm in STORMS:
        # Submissions
        sub_path = INPUT_DIR / f"{storm}_submissions.parquet"
        if sub_path.exists():
            df = pd.read_parquet(sub_path)
            df_general = df[df["subreddit"].isin(GENERAL_SUBREDDITS)]
            sample = df_general.sample(
                min(len(df_general), n_per_storm), random_state=42
            )
            all_subs.append(sample)
            print(f"  {storm} submissions: {len(df_general)} general "
                  f"→ sampled {len(sample)}")

        # Comments
        com_path = INPUT_DIR / f"{storm}_comments.parquet"
        if com_path.exists():
            df = pd.read_parquet(com_path)
            df_general = df[df["subreddit"].isin(GENERAL_SUBREDDITS)]
            sample = df_general.sample(
                min(len(df_general), n_per_storm), random_state=42
            )
            all_coms.append(sample)
            print(f"  {storm} comments:    {len(df_general)} general "
                  f"→ sampled {len(sample)}")

    # Save
    if all_subs:
        df_out = (pd.concat(all_subs, ignore_index=True)
                    .sort_values(["storm", "subreddit", "created_utc"]))
        out = OUTPUT_DIR / "general_subreddits_submissions_sample.csv"
        df_out.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\nSaved {len(df_out)} rows → {out.name}")

    if all_coms:
        df_out = (pd.concat(all_coms, ignore_index=True)
                    .sort_values(["storm", "subreddit", "created_utc"]))
        out = OUTPUT_DIR / "general_subreddits_comments_sample.csv"
        df_out.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"Saved {len(df_out)} rows → {out.name}")

    print("\nInspect these CSVs in Excel before proceeding to Step 2.")
    print("Check: are all posts actually hurricane-relevant?")
    print("If yes, run extract_keywords() below.")


# Part B: TF-IDF + KeyBERT keyword extraction
## Only run this after inspecting the samples above and confirming quality

def extract_keywords():
    """
    Use TF-IDF and KeyBERT to derive hurricane keywords empirically
    from the verified general subreddit content.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from keybert import KeyBERT
    import numpy as np
    import random

    print("Loading corpus...\n")

    # Build positive corpus (general subreddits)
    positive_texts = []
    background_texts = []

    for storm in STORMS:
        sub_path = INPUT_DIR / f"{storm}_submissions.parquet"
        if not sub_path.exists():
            continue
        df = pd.read_parquet(sub_path)

        # Positive: general subreddits -> hurricane-specific by definition
        positive_texts.extend(
            df[df["subreddit"].isin(GENERAL_SUBREDDITS)]["text_clean"].dropna().tolist()
        )

        # Background: location subreddits -> mixed content
        background_texts.extend(
            df[~df["subreddit"].isin(GENERAL_SUBREDDITS)]["text_clean"].dropna().tolist()
        )

    print(f"Positive corpus (general subreddits): {len(positive_texts)} documents")
    print(f"Background corpus (location subreddits): {len(background_texts)} documents\n")

    # TF-IDF: find words distinctive to hurricane content
    print("Running TF-IDF...")

    all_texts  = positive_texts + background_texts
    all_labels = [1] * len(positive_texts) + [0] * len(background_texts)

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),      # unigrams and bigrams
        stop_words="english",
        min_df=5,                # must appear in at least 5 docs
        max_df=0.8,              # ignore very common words
    )
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    feature_names = vectorizer.get_feature_names_out()

    # Average TF-IDF score per class
    pos_idx = [i for i, l in enumerate(all_labels) if l == 1]
    neg_idx = [i for i, l in enumerate(all_labels) if l == 0]

    pos_scores = np.asarray(tfidf_matrix[pos_idx].mean(axis=0)).flatten()
    neg_scores = np.asarray(tfidf_matrix[neg_idx].mean(axis=0)).flatten()

    # Distinctiveness = high in positive, low in background
    distinctiveness = pos_scores - neg_scores

    df_tfidf = pd.DataFrame({
        "term":            feature_names,
        "pos_score":       pos_scores,
        "neg_score":       neg_scores,
        "distinctiveness": distinctiveness,
    }).sort_values("distinctiveness", ascending=False)

    # Save full TF-IDF results for inspection
    df_tfidf.to_csv(OUTPUT_DIR / "tfidf_keywords.csv", index=False, encoding="utf-8-sig")
    print(f"Top 30 TF-IDF distinctive terms:")
    print(df_tfidf.head(30)[["term", "distinctiveness"]].to_string(index=False))

    # KeyBERT: semantic keyword extraction
    print("\nRunning KeyBERT ...")

    # Use a sample of positive texts to keep it manageable
    random.seed(42)
    sample_size = min(1000, len(positive_texts))
    sample_texts = random.sample(positive_texts, sample_size)
    combined_text = " ".join(sample_texts)

    kw_model = KeyBERT()
    keybert_keywords = kw_model.extract_keywords(
        combined_text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=50,
        diversity=0.5,           # penalise redundant keywords
    )

    df_keybert = pd.DataFrame(keybert_keywords, columns=["term", "score"])
    df_keybert.to_csv(OUTPUT_DIR / "keybert_keywords.csv", index=False, encoding="utf-8-sig")
    print(f"\nTop 30 KeyBERT keywords:")
    print(df_keybert.head(30).to_string(index=False))

    # Combined list
    # Take top 50 from TF-IDF + top 50 from KeyBERT, deduplicate
    combined = sorted(set(df_tfidf.head(50)["term"].tolist()) |
                      set(df_keybert.head(50)["term"].tolist()))
    pd.DataFrame({"keyword": combined}).to_csv(
        OUTPUT_DIR / "combined_keywords_for_review.csv",
        index=False, encoding="utf-8-sig",
    )
    print(f"\nCombined keyword list: {len(combined)} terms saved to combined_keywords_for_review.csv")
    print("Next: prune over-broad terms, then update STRONG_KEYWORDS in config/settings.py.")    


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Keyword derivation for hurricane filtering.")
    parser.add_argument("--part", choices=["A", "B", "both"], default="A",
                        help="A = sample inspection only, B = TF-IDF + KeyBERT, both = run A then B")
    args = parser.parse_args()
 
    if args.part in ("A", "both"):
        sample_general_subreddits(n_per_storm=50)
    if args.part in ("B", "both"):
        extract_keywords()