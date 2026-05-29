"""
- Run Steps 2a and 2b: 
    - time-filter Reddit data
    - apply keyword filtering
 
- Requires:
    - data/raw/reddit/subreddits.zip  (Academic Torrents download)
    - data/processed/hurricane_data/hurricane_archive_links.csv  (from Step 1a)
 
- Output:
    - data/processed/reddit/{Storm}_comments.parquet
    - data/processed/reddit/{Storm}_submissions.parquet
    - data/processed/reddit/filtered/{Storm}_*_filtered.parquet
"""
 
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
 
from src.data_processing.reddit_data.reddit_preprocessing import run as run_preprocess
from src.data_processing.reddit_data.reddit_filtering import run as run_filter
 
if __name__ == "__main__":
    print("=" * 60)
    print("Step 2a: Time-filtering Reddit data by storm window")
    print("=" * 60)
    run_preprocess()
 
    print("\n" + "=" * 60)
    print("Step 2b: Keyword filtering and cleaning")
    print("=" * 60)
    run_filter()