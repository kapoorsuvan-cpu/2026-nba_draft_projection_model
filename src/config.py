from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

for directory in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, REPORTS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Raw files are now API caches, not required manual inputs. If a cache exists it is reused;
# otherwise src/data_sources.py fetches from the configured APIs and writes the cache.
RAW_FILES = {
    "college": RAW_DIR / "historical_college_stats.csv",
    "draft": RAW_DIR / "historical_draft_results.csv",
    "nba": RAW_DIR / "historical_nba_outcomes.csv",
    "recruiting": RAW_DIR / "recruiting_rankings.csv",
    "espn_2026": RAW_DIR / "espn_2026_top100_raw.csv",
}

INTERIM_FILES = {
    "college_clean": INTERIM_DIR / "historical_player_seasons_clean.csv",
    "final_seasons": INTERIM_DIR / "final_college_season_players.csv",
    "nba_clean": INTERIM_DIR / "nba_outcomes_clean.csv",
    "recruiting_clean": INTERIM_DIR / "recruiting_rankings_clean.csv",
    "prospects_clean": INTERIM_DIR / "2026_prospects_clean.csv",
}

PROCESSED_FILES = {
    "training": PROCESSED_DIR / "model_training_dataset.csv",
    "training_labeled": PROCESSED_DIR / "model_training_dataset_with_labels.csv",
    "prospect_features": PROCESSED_DIR / "2026_draft_class_features.csv",
    "prospect_predictions": PROCESSED_DIR / "2026_draft_class_predictions.csv",
    "feature_dictionary": PROCESSED_DIR / "feature_dictionary.csv",
    "unmatched": PROCESSED_DIR / "unmatched_players_report.csv",
}

# Data acquisition: API-first. CSVs are used only as local caches/fallbacks.
USE_API_DATA_SOURCES = True
REFRESH_API_CACHE = False

# Free API keys / packages used by the project:
# 1) CollegeBasketballData / cbbd: free key, stored in CBBD_API_KEY or BEARER_TOKEN.
# 2) nba_api: free Python package for NBA.com stats; no key required.
CBBD_API_KEY_ENV = "CBBD_API_KEY"
CBBD_ALT_API_KEY_ENV = "BEARER_TOKEN"
CBBD_API_HOST = "https://api.collegebasketballdata.com"
NBA_API_SLEEP_SECONDS = 0.65
CBBD_API_SLEEP_SECONDS = 0.35

LABEL_THRESHOLDS = {
    "rotation_min_qualifying_seasons": 2,
    "rotation_min_post_rookie_qualifying_seasons": 1,
    "rotation_min_games_per_season": 40,
    "rotation_min_minutes_per_game": 15,
}

TARGET_COLUMN = "nba_success_label"
CLASS_ORDER = ["Not NBA Level", "Rotation", "Star"]

# A Rotation label requires an observed season in NBA year five or later.
# The 2021 class completed year five in 2025-26; the 2022 class has not.
ELIGIBLE_DRAFT_MIN_YEAR = 2006
ELIGIBLE_DRAFT_MAX_YEAR = 2021
TRAIN_MAX_DRAFT_YEAR = 2019
TEST_MIN_DRAFT_YEAR = 2020
TEST_MAX_DRAFT_YEAR = 2021
API_DRAFT_YEARS = list(range(ELIGIBLE_DRAFT_MIN_YEAR, ELIGIBLE_DRAFT_MAX_YEAR + 1))
API_COLLEGE_SEASONS = list(range(ELIGIBLE_DRAFT_MIN_YEAR, ELIGIBLE_DRAFT_MAX_YEAR + 1))
API_RECRUITING_YEARS = list(range(2013, ELIGIBLE_DRAFT_MAX_YEAR + 1))

RUN_2026_PROSPECT_PIPELINE_BY_DEFAULT = False
MIN_POSITION_SAMPLE = 40
MIN_POSITION_MODEL_SAMPLE = 10
RANDOM_STATE = 42

SPECIFIC_POSITIONS = ["G", "F", "C"]
POSITION_GROUPS = ["guard", "wing", "big"]

OUTCOME_COLUMNS = {
    "nba_games_played_first4", "nba_games_started_first4", "nba_minutes_first4",
    "nba_minutes_per_game_first4", "nba_peak_minutes_per_game_first4",
    "nba_points_per_game_first4", "nba_rebounds_per_game_first4", "nba_assists_per_game_first4",
    "nba_win_shares_first4", "nba_vorp_first4", "nba_bpm_first4", "nba_ws_per_48_first4",
    "seasons_played_first4", "second_contract_indicator", "all_star_indicator",
    "all_nba_indicator", "max_contract_indicator", "nba_games_played_career",
    "nba_minutes_career", "seasons_played_career", "qualifying_rotation_seasons_career",
    "qualifying_rotation_seasons_year5_plus", "nba_success_score", "nba_success_label", "is_star",
    "is_rotation_or_better", "is_not_nba_level"
}

IDENTIFIER_COLUMNS = [
    "player_name", "normalized_player_name", "college_team", "conference", "ncaa_season",
    "draft_year", "nba_team_drafted_by", "draft_round", "draft_pick_overall", "position",
    "primary_position", "secondary_position", "position_group", "height", "weight", "birth_date",
    "age_final_college_season", "class_year", "recruiting_rank", "recruiting_stars", "recruiting_source"
]
