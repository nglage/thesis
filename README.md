# Tracking Hurricane Risk Perception
### Nora Grace Glage - University of Copenhagen

This repository contains all code used for data collection, preprocessing, 
scoring, and analysis for the master's thesis "Tracking Hurricane Risk 
Perception." The thesis examines how Reddit discourse responds to official 
NHC advisory information across six Atlantic hurricanes (Harvey, Irma, 
Dorian, Ida, Ian, Milton) using a newly constructed Risk Perception Index (RPI).

Raw Reddit data is not included due to privacy considerations but can be 
obtained as described below.

## Methods

Risk-related statements are classified with zero-shot natural language inference using a pretrained transformer model, alongside a supervised emotion classifier over the same corpus. NHC advisories are scraped from the official archive and aligned with the Reddit timeline for analysis. Label quality is sense-checked against the theoretical risk perception dimensions the index is built on (`rpi_sensecheck.py`).

## Setup

1. Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

2. Copy the paths template and fill in your local directories:
```bash
cp paths.py.template paths.py
```

3. Obtain Reddit data from the Academic Torrents archive of historical 
Reddit dumps:

https://academictorrents.com/details/3e3f64dee22dc304cdd2546254ca1f8e8ae542b4

Download the relevant subreddits (listed in `config/settings.py`) and place them in the directory specified in `paths.py`.

## Running Pipeline
```bash
python scripts/run_all.py                  # Full pipeline
python scripts/run_all.py --skip-nhc       # Skip NHC scraping
python scripts/run_all.py --analysis-only  # Steps 4-5 only
```

NHC advisories are scraped automatically. Manual label validation can be run separately:
```bash
python src/data_processing/risk_score/rpi_sensecheck.py
```
