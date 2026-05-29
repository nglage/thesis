"""
- Template for local path configuration
- Copy to paths.py (gitignored) and fill in the absolute local paths
 
- Usage in scripts:
    from paths import PATHS
"""

from pathlib import Path

PATHS = {
    # Raw Reddit ZIP downloaded from Academic Torrents
    "reddit_zip": Path(r"FILL_IN/subreddits.zip"),
 
    # All processed data is under this root
    "data_processed": Path(r"FILL_IN/data/processed"),
 
    # Output directories for figures and tables
    "figures": Path(r"FILL_IN/figures"),
    "tables":  Path(r"FILL_IN/tables"),
}