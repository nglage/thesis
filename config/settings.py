"""
- Central configuration for all project-wide constants
- Import from here in every script instead of hard-coding values
"""

# Storms
STORMS = ["Harvey", "Irma", "Dorian", "Ida", "Ian", "Milton"]
 
STORM_YEARS = {
    "Harvey": 2017,
    "Irma":   2017,
    "Dorian": 2019,
    "Ida":    2021,
    "Ian":    2022,
    "Milton": 2024,
}
 
STORM_COLORS = {
    "Harvey": "#E63946",
    "Irma":   "#457B9D",
    "Dorian": "#2A9D8F",
    "Ida":    "#E9C46A",
    "Ian":    "#F4A261",
    "Milton": "#9B5DE5",
}

# Reddit Subreddit mapping

SUBREDDIT_STORM_MAP = {

    # General hurricane subreddits —> all storms
    "hurricane":       ["Harvey", "Irma", "Dorian", "Ida", "Ian", "Milton"],
    "TropicalWeather": ["Harvey", "Irma", "Dorian", "Ida", "Ian", "Milton"],
   
    # Texas/Gulf Coast
    "texas":           ["Harvey", "Ida"],
    "houston":         ["Harvey", "Ida"],

    # Louisiana
    "Louisiana":       ["Harvey", "Ida"],
    "NewOrleans":      ["Harvey", "Ida"],
    "batonrouge":      ["Harvey", "Ida"],

    # Florida
    "florida":         ["Irma", "Dorian", "Ian", "Milton"],
    "Miami":           ["Irma", "Dorian"],
    "FortMyers":       ["Irma", "Ian"],
    "tampa":           ["Irma", "Ian", "Milton"],
    "orlando":         ["Irma", "Ian", "Milton"],

    # Carolinas
    "NorthCarolina":   ["Dorian", "Ian"],
    "southcarolina":   ["Dorian", "Ian"],

    # Northeast (Ida only)
    "newjersey":       ["Ida"],
    "nyc":             ["Ida"],
    "philadelphia":    ["Ida"],
}

GENERAL_SUBREDDITS = {"hurricane", "TropicalWeather"}

SUBREDDIT_TYPE_MAP = {
    "hurricane":       "general_weather",
    "TropicalWeather": "general_weather",
    "texas":           "state",
    "florida":         "state",
    "Louisiana":       "state",
    "NorthCarolina":   "state",
    "southcarolina":   "state",
    "newjersey":       "state",
    "houston":         "local_community",
    "NewOrleans":      "local_community",
    "batonrouge":      "local_community",
    "tampa":           "local_community",
    "orlando":         "local_community",
    "Miami":           "local_community",
    "FortMyers":       "local_community",
    "nyc":             "local_community",
    "philadelphia":    "local_community",
}

PROXIMITY_TYPES = {"state", "local_community"}

# Reddit preprocessing
PRE_BUFFER_DAYS  = 3   # days before first NHC advisory to include
POST_BUFFER_DAYS = 7   # days after last NHC advisory to include

# Reddit filtering
STORM_NAME_PATTERNS = ["harvey", "irma", "dorian", "ida", "ian", "milton"]
 
STRONG_KEYWORDS = {
    "hurricane", "tropical storm", "tropical depression", "tropical cyclone", # storm types
    "nhc", "national hurricane center", "forecast",
    "hurricane watch", "hurricane warning", "tropical storm watch",
    "tropical storm warning", # nhc/offical information
    "landfall", "eyewall", "eye wall", "storm surge", "storm track",
    "category", "wind speed", "mph winds", "knots", "millibars", "pressure drop",
    "sustained winds", "maximum sustained", "major hurricane", # track/intensity
    "rainfall", "tornado warning", "storm band", # weather phenomena
    "mandatory evacuation", "evacuation order", "evacuation zone",
    "hurricane shelter", "storm prep", "hurricane supplies", # response & preparedness
    "storm flooding", "flood warning", "flood zone", "storm damage",
    "power outage", "power restoration",
    "state of emergency", "national guard",
    "coast guard", "red cross" # official response
}

# NLI scoring
NLI_MODEL    = "facebook/bart-large-mnli"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

## NLI labels
LABEL_FEAR       = "Expressing fear, worry or anxiety about the storm"
LABEL_DISMISSIVE = "Dismissing or downplaying the storm threat"
LABEL_LIKELIHOOD = "Making specific predictions or probability assessments about the storm's track or intensity"
LABEL_UNCERTAINTY= "Expressing doubt or uncertainty about where or whether the storm will make landfall or affect a specific area"
LABEL_VULNERABLE = "Describing direct personal exposure to the storm, such as being in the storm's path, having to evacuate, or experiencing storm impacts firsthand"
LABEL_PROTECTIVE = "Discussing evacuation or protective action taken or planned"
LABEL_FACTUAL    = "Sharing factual or official information about the storm"

DISCOURSE_LABELS = [
    LABEL_FEAR, LABEL_DISMISSIVE, LABEL_LIKELIHOOD,
    LABEL_UNCERTAINTY, LABEL_VULNERABLE, LABEL_PROTECTIVE, LABEL_FACTUAL,
]

NLI_RELEVANCE_THRESHOLD = 0.70  # = minimum probability(hurricane relevance) to retain post
SCORING_BATCH_SIZE = 128
MAX_TEXT_CHARS = 2000
LOG_EPSILON = 0.001  # for RPI_A_log = log(fear + epsilon)

# RPI composite
'''
- weights are variance-based and estimated empirically in rpi_scoring.py
- values below derived from previous full corpus run
- overwritten at runtime if recalculated
'''
RPI_WEIGHTS_DEFAULT = {"w_A": 0.31, "w_D": 0.37, "w_E": 0.33}

# Regression
MIN_POSTS_PER_HOUR = 5  # minimum posts per storm-hour to include in hourly dataset
ROLLING_WINDOW = 6  # hours for backward rolling mean in time series plots

## Advisory predictors in regression
ADVISORY_PREDS = [
    "delta_forecast_1_wind_z",
    "delta_wind_kt_z",
    "delta_fc_lag1_z",
    "delta_wind_lag1_z",
    "new_hurricane_warning",
    "warning_lag1",
    "new_hurricane_watch",
]

## outcome columns for hourly regression
OUTCOME_COLS = {
    "mean_rpi_composite": ("RPI_composite", "mean"),
    "mean_rpi_composite_full": ("RPI_composite_full", "mean"),
    "mean_rpi_a": ("RPI_A", "mean"),
    "mean_rpi_d": ("RPI_D_rev", "mean"),
    "mean_rpi_e": ("RPI_E_rev", "mean"),
    "mean_rpi_simple": ("RPI_simple", "mean"),
    "mean_p_likelihood": ("p_likelihood", "mean"),
    "mean_p_dismissive": ("p_dismissive", "mean"),
    "mean_p_vulnerable": ("p_vulnerable", "mean"),
    "mean_p_protective": ("p_protective", "mean"),
    "mean_fear": ("fear", "mean"),
    "n_posts": ("RPI_composite", "count"),
    "hour_sin": ("hour_sin", "first"),
    "hour_cos": ("hour_cos", "first"),
}
