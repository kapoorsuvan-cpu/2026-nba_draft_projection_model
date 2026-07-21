import logging
import numpy as np
import pandas as pd
from .config import OUTCOME_COLUMNS, IDENTIFIER_COLUMNS, PROCESSED_FILES
from .utils import write_csv

logger = logging.getLogger(__name__)


def _safe_div(num, den):
    return np.where((pd.notna(den)) & (den != 0), num / den, np.nan)


def create_engineered_features(df: pd.DataFrame, save_dictionary: bool = False) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if col not in {"player_name", "normalized_player_name", "college_team", "conference", "position", "primary_position", "secondary_position", "position_group",
        "analysis_position", "class_year", "birth_date", "recruiting_source", "missing_stats_reason"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    mpg = pd.to_numeric(df.get("minutes_per_game"), errors="coerce")
    age = pd.to_numeric(df.get("age_final_college_season", df.get("age")), errors="coerce")
    usage = pd.to_numeric(df.get("usage_rate"), errors="coerce")
    ppg = pd.to_numeric(df.get("points_per_game"), errors="coerce")
    rpg = pd.to_numeric(df.get("rebounds_per_game"), errors="coerce")
    apg = pd.to_numeric(df.get("assists_per_game"), errors="coerce")
    spg = pd.to_numeric(df.get("steals_per_game"), errors="coerce")
    bpg = pd.to_numeric(df.get("blocks_per_game"), errors="coerce")
    tpg = pd.to_numeric(df.get("turnovers_per_game"), errors="coerce")
    bpm = pd.to_numeric(df.get("box_plus_minus"), errors="coerce")

    df["points_per_40"] = _safe_div(ppg * 40, mpg)
    df["rebounds_per_40"] = _safe_div(rpg * 40, mpg)
    df["assists_per_40"] = _safe_div(apg * 40, mpg)
    df["steals_per_40"] = _safe_div(spg * 40, mpg)
    df["blocks_per_40"] = _safe_div(bpg * 40, mpg)
    df["turnovers_per_40"] = _safe_div(tpg * 40, mpg)
    df["stock_rate_per_40"] = df["steals_per_40"] + df["blocks_per_40"]
    df["assist_to_turnover_ratio"] = _safe_div(apg, tpg)

    df["age_adjusted_ppg"] = ppg - 0.75 * (age - 19)
    df["age_adjusted_usage"] = usage - 1.5 * (age - 19)
    df["age_adjusted_bpm"] = bpm - 0.5 * (age - 19)

    ts = pd.to_numeric(df.get("true_shooting_percentage"), errors="coerce")
    efg = pd.to_numeric(df.get("effective_field_goal_percentage"), errors="coerce")
    ft = pd.to_numeric(df.get("free_throw_percentage"), errors="coerce")
    three = pd.to_numeric(df.get("three_point_percentage"), errors="coerce")
    two = pd.to_numeric(df.get("two_point_percentage"), errors="coerce")
    df["scoring_efficiency_index"] = pd.concat([ts, efg, ft], axis=1).mean(axis=1)
    df["shooting_translation_score"] = 0.40 * three + 0.25 * ft + 0.25 * efg + 0.10 * pd.to_numeric(df.get("three_point_attempt_rate"), errors="coerce")
    df["defensive_event_score"] = 0.55 * df["steals_per_40"] + 0.45 * df["blocks_per_40"]
    df["guard_skill_score"] = 0.45 * df["assists_per_40"] + 0.25 * df["assist_to_turnover_ratio"] + 0.30 * df["shooting_translation_score"]
    df["wing_skill_score"] = 0.35 * df["shooting_translation_score"] + 0.30 * df["stock_rate_per_40"] + 0.20 * df["points_per_40"] + 0.15 * pd.to_numeric(df.get("defensive_rebound_rate"), errors="coerce")
    df["big_skill_score"] = 0.40 * df["blocks_per_40"] + 0.30 * pd.to_numeric(df.get("total_rebound_rate"), errors="coerce") + 0.20 * two + 0.10 * df["points_per_40"]

    class_year = df.get("class_year", pd.Series(index=df.index, dtype=object)).astype(str).str.lower()
    df["is_one_and_done"] = class_year.isin(["fr", "freshman", "1", "one-and-done"]).astype(int)
    df["is_upperclassman"] = class_year.isin(["jr", "sr", "junior", "senior", "3", "4"]).astype(int)
    conf = df.get("conference", pd.Series(index=df.index, dtype=object)).astype(str).str.upper()
    high_major = {"ACC", "SEC", "BIG 12", "BIG12", "BIG TEN", "BIG10", "B1G", "PAC-12", "PAC12", "BIG EAST", "BIGEAST"}
    df["is_high_major"] = conf.isin(high_major).astype(int)
    rank = pd.to_numeric(df.get("recruiting_rank"), errors="coerce")
    df["log_recruiting_rank"] = np.log1p(rank)

    if "draft_pick_overall" in df.columns:
        pick = pd.to_numeric(df["draft_pick_overall"], errors="coerce")
        df["draft_pick_tier"] = pd.cut(pick, bins=[0, 5, 14, 30, 45, 60, np.inf], labels=["top5", "lottery", "first_round", "early_second", "late_second", "undrafted_or_unknown"]).astype(str)

    for col in ["has_college_stats", "has_recruiting_rank", "is_international_or_pro"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    if save_dictionary:
        feature_dict = pd.DataFrame({
            "feature": [c for c in df.columns if c not in OUTCOME_COLUMNS],
            "description": ["Generated or cleaned model input feature" for _ in range(len([c for c in df.columns if c not in OUTCOME_COLUMNS]))]
        })
        write_csv(feature_dict, PROCESSED_FILES["feature_dictionary"])
    return df


def get_feature_columns(df: pd.DataFrame, include_draft_pick: bool = True, include_position: bool = True) -> list[str]:
    leakage = set(OUTCOME_COLUMNS) | {"target", "label"}
    exclude = leakage | {"player_name", "normalized_player_name", "college_team", "normalized_college_team", "nba_team_drafted_by", "birth_date", "recruiting_source", "missing_stats_reason", "notes", "key_positive_features", "key_negative_features"}
    if not include_draft_pick:
        exclude |= {"draft_pick_overall", "draft_round", "draft_pick_tier"}
    if not include_position:
        exclude |= {"position", "primary_position", "secondary_position", "position_group"}
    return [c for c in df.columns if c not in exclude]

# Override feature selection with leakage-safe version.
def get_feature_columns(
    df,
    include_draft_pick=True,
    include_position=True,
    include_identifiers=False,
):
    """
    Return model feature columns while preventing NBA-outcome leakage.

    Allowed features should be known before/during the draft:
    - college production
    - age/class
    - recruiting
    - position/physical profile
    - team/context
    - draft pick, only if include_draft_pick=True

    Excluded:
    - NBA outcomes
    - labels
    - player IDs/names
    - first-four-season NBA stats
    - second-contract/all-star/all-nba indicators
    """

    leakage_patterns = [
        "nba_",
        "_nba",
        "first4",
        "first_4",
        "all_star",
        "all_nba",
        "second_contract",
        "success",
        "label",
        "is_star",
        "is_rotation",
        "is_not_nba",
        "seasons_15mpg",
        "seasons_played",
        "win_shares",
        "vorp",
        "bpm_nba",
        "games_started_first",
        "games_played_first",
        "minutes_first",
        "peak_minutes",
    ]

    identifier_patterns = [
        "player_name",
        "normalized_player_name",
        "person_id",
        "athlete_id",
        "team_abbreviation",
        "organization_type",
        "match_score",
        "match_scor",
    ]

    draft_cols = {
        "draft_year",
        "draft_year_college",
        "ncaa_season",
        "draft_round",
        "draft_pick_overall",
        "draft_round_pick",
        "draft_pick_tier",
    }

    allowed_object_cols = {
        "position",
        "primary_position",
        "secondary_position",
        "position_group",
        "analysis_position",
        "college_team",
        "conference",
        "class_year",
        "recruiting_source",
    }

    feature_cols = []

    for col in df.columns:
        c = str(col)
        lc = c.lower()

        if any(p in lc for p in leakage_patterns):
            continue

        if not include_identifiers and any(p in lc for p in identifier_patterns):
            continue

        if not include_draft_pick and c in draft_cols:
            continue

        if not include_position and c in {
            "position",
            "primary_position",
            "secondary_position",
            "position_group",
        "analysis_position",
        }:
            continue

        if c in {
            "nba_success_label",
            "nba_success_score",
            "analysis_position",
            "is_star",
            "is_rotation_or_better",
            "is_not_nba_level",
        }:
            continue

        # Keep numeric features.
        if c in df.select_dtypes(include=["number", "bool"]).columns:
            feature_cols.append(c)
            continue

        # Keep only approved categorical pre-draft features.
        if c in allowed_object_cols:
            feature_cols.append(c)

    # Remove columns that are entirely missing.
    usable = []
    for col in feature_cols:
        if col in df.columns and df[col].notna().sum() > 0:
            usable.append(col)

    return sorted(set(usable))

# FINAL CLEAN COLLEGE-ONLY FEATURE SELECTION OVERRIDE


# FINAL CLEAN COLLEGE-ONLY FEATURE SELECTION OVERRIDE
def get_feature_columns(
    df,
    include_draft_pick=True,
    include_position=True,
    include_identifiers=False,
):
    """
    Leakage-safe and noise-reduced feature selection.

    Main goal:
    - Allow college/recruiting/physical/context features.
    - Exclude NBA outcome columns.
    - Exclude draft-pick columns unless explicitly requested.
    - Exclude merge artifacts, IDs, names, raw years, and noisy duplicate columns.
    """

    leakage_patterns = [
        "nba_",
        "_nba",
        "first4",
        "first_4",
        "all_star",
        "all_nba",
        "second_contract",
        "success",
        "label",
        "is_star",
        "is_rotation",
        "is_not_nba",
        "seasons_15mpg",
        "seasons_played",
        "win_shares",
        "vorp",
        "bpm_nba",
        "games_started_first",
        "games_played_first",
        "minutes_first",
        "peak_minutes",
    ]

    identifier_or_artifact_patterns = [
        "player_name",
        "normalized_player_name",
        "person_id",
        "athlete_id",
        "team_abbreviation",
        "organization_type",
        "match_score",
        "match_scor",
        "_left_row_id",
        "_right_row_id",
    ]

    noisy_exact_cols = {
        # Targets / labels
        "nba_success_label",
        "nba_success_score",
        "is_star",
        "is_rotation_or_better",
        "is_not_nba_level",

        # Raw or duplicate identity fields
        "player_nam",
        "normalized_player_nam",
        "player",
        "name",

        # Position helper columns that should not be predictive inputs
        "analysis_position",
        "primary_position",
        "secondary_position",
        "position_group",
        "position_recruiting",
        "positio",

        # Year columns that can act like timeline/draft leakage
        "draft_year",
        "draft_year_college",
        "ncaa_season",
        "recruiting_year",

        # Merge/debug artifacts
        "match_status",
        "merge_name",

        # Mostly-empty or unstable API fields
        "organization_type",
        "conference_college",
        "college_team_college",
    }

    draft_cols = {
        "draft_round",
        "draft_pick_overall",
        "draft_round_pick",
        "draft_pick_tier",
    }

    allowed_categorical_cols = {
        "position",
        "conference",
        "class_year",
    }

    if include_position:
        allowed_categorical_cols.add("position")

    feature_cols = []

    numeric_cols = set(df.select_dtypes(include=["number", "bool"]).columns)

    for col in df.columns:
        c = str(col)
        lc = c.lower()

        # OUTCOME_COLUMNS is the authoritative boundary. Pattern checks below
        # are defense in depth for unexpected API/merge variants.
        if c in OUTCOME_COLUMNS:
            continue

        if c in noisy_exact_cols:
            continue

        if any(pattern in lc for pattern in leakage_patterns):
            continue

        if not include_identifiers and any(pattern in lc for pattern in identifier_or_artifact_patterns):
            continue

        if not include_draft_pick and c in draft_cols:
            continue

        if not include_position and c in {"position", "primary_position", "secondary_position", "position_group", "analysis_position"}:
            continue

        # Keep numeric basketball/recruiting/physical features.
        if c in numeric_cols:
            feature_cols.append(c)
            continue

        # Keep only approved categorical pre-draft fields.
        if c in allowed_categorical_cols:
            feature_cols.append(c)

    # Remove entirely missing or constant columns.
    usable = []
    for col in feature_cols:
        if col not in df.columns:
            continue

        s = df[col]

        if s.notna().sum() == 0:
            continue

        if s.nunique(dropna=True) <= 1:
            continue

        usable.append(col)

    return sorted(set(usable))
