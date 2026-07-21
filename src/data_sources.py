"""API-first data acquisition for the NBA college projection model.

This module uses two free sources:
- CollegeBasketballData / cbbd for NCAA player stats, team context, recruiting, and
  optionally draft data. Requires a free bearer token in CBBD_API_KEY or BEARER_TOKEN.
- nba_api for NBA.com draft history and player career stats. No API key required.

CSV files in data/raw/ are treated as caches. Delete them or set REFRESH_API_CACHE=True
in config.py to re-fetch from the APIs.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from .config import (
    API_COLLEGE_SEASONS,
    API_DRAFT_YEARS,
    API_RECRUITING_YEARS,
    CBBD_ALT_API_KEY_ENV,
    CBBD_API_HOST,
    CBBD_API_KEY_ENV,
    CBBD_API_SLEEP_SECONDS,
    NBA_API_SLEEP_SECONDS,
    RAW_FILES,
    REFRESH_API_CACHE,
    USE_API_DATA_SOURCES,
)
from .utils import safe_read_csv, write_csv

logger = logging.getLogger(__name__)


def _cache_or_fetch(path: Path, fetcher, required: bool = True) -> pd.DataFrame:
    if path.exists() and not REFRESH_API_CACHE:
        logger.info("Loading cached raw data: %s", path)
        return safe_read_csv(path, required=required)
    if not USE_API_DATA_SOURCES:
        return safe_read_csv(path, required=required)
    df = fetcher()
    if df.empty and path.exists():
        logger.warning("API fetch returned no rows; falling back to existing cache: %s", path)
        return safe_read_csv(path, required=required)
    if df.empty and required:
        raise RuntimeError(f"Unable to fetch required data and no cache exists: {path}")
    write_csv(df, path)
    return df


def _records_to_df(records: Any) -> pd.DataFrame:
    """Convert cbbd/nba_api/model objects/lists/dicts to a DataFrame."""
    if records is None:
        return pd.DataFrame()
    if isinstance(records, pd.DataFrame):
        return records.copy()
    if hasattr(records, "get_data_frames"):
        frames = records.get_data_frames()
        return frames[0] if frames else pd.DataFrame()
    if hasattr(records, "to_dict") and not isinstance(records, list):
        records = records.to_dict()
    if isinstance(records, dict):
        # Some clients return {"data": [...]} or a raw object dict.
        for key in ["data", "items", "results"]:
            if key in records and isinstance(records[key], list):
                records = records[key]
                break
        else:
            records = [records]
    rows = []
    for item in records if isinstance(records, list) else [records]:
        if hasattr(item, "to_dict"):
            rows.append(item.to_dict())
        elif isinstance(item, dict):
            rows.append(item)
        else:
            rows.append(vars(item) if hasattr(item, "__dict__") else {"value": item})
    return pd.json_normalize(rows) if rows else pd.DataFrame()


def _clean_api_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_").replace(".", "_") for c in out.columns]
    return out


def _cbbd_key() -> str | None:
    return os.getenv(CBBD_API_KEY_ENV) or os.getenv(CBBD_ALT_API_KEY_ENV)


def _require_cbbd_key() -> str:
    key = _cbbd_key()
    if not key:
        raise RuntimeError(
            "Missing CollegeBasketballData API key. Get a free key at "
            "https://collegebasketballdata.com/key and set CBBD_API_KEY=your_key "
            "or BEARER_TOKEN=your_key."
        )
    return key


def _cbbd_rest_get(endpoint: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    key = _require_cbbd_key()
    url = f"{CBBD_API_HOST.rstrip('/')}/{endpoint.lstrip('/')}"
    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
    response = requests.get(url, params={k: v for k, v in (params or {}).items() if v is not None}, headers=headers, timeout=60)
    if response.status_code >= 400:
        raise RuntimeError(f"CBBD request failed {response.status_code}: {url} {response.text[:500]}")
    time.sleep(CBBD_API_SLEEP_SECONDS)
    return _clean_api_columns(_records_to_df(response.json()))


def _fetch_cbbd_by_year(endpoint: str, years: Iterable[int], extra_params: dict[str, Any] | None = None) -> pd.DataFrame:
    frames = []
    for year in years:
        logger.info("Fetching CBBD %s for season/year %s", endpoint, year)
        params = dict(extra_params or {})
        # CBBD docs use `season` for stats endpoints and `year` for recruiting.
        if endpoint.startswith("/recruiting"):
            params["year"] = year
        else:
            params["season"] = year
        try:
            df = _cbbd_rest_get(endpoint, params)
            if not df.empty:
                frames.append(df)
        except Exception as exc:
            logger.warning("CBBD fetch failed for %s %s: %s", endpoint, year, exc)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_historical_college_stats() -> pd.DataFrame:
    """Fetch NCAA player-season stats and enrich with shooting/team context from CBBD."""
    base = _fetch_cbbd_by_year("/stats/player/season", API_COLLEGE_SEASONS)
    if base.empty:
        return base
    base = _normalize_cbbd_player_stats(base)

    shooting = _fetch_cbbd_by_year("/stats/player/shooting/season", API_COLLEGE_SEASONS)
    if not shooting.empty:
        shooting = _normalize_cbbd_shooting_stats(shooting)
        merge_keys = [k for k in ["ncaa_season", "athlete_id", "player_name", "college_team"] if k in base.columns and k in shooting.columns]
        if merge_keys:
            base = base.merge(shooting, on=merge_keys, how="left", suffixes=("", "_shooting"))

    teams = _fetch_cbbd_by_year("/stats/team/season", API_COLLEGE_SEASONS)
    if not teams.empty:
        teams = _normalize_cbbd_team_stats(teams)
        merge_keys = [k for k in ["ncaa_season", "college_team"] if k in base.columns and k in teams.columns]
        if merge_keys:
            base = base.merge(teams, on=merge_keys, how="left", suffixes=("", "_team"))

    return base


def _normalize_cbbd_player_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = _clean_api_columns(df)
    out = pd.DataFrame()
    out["player_name"] = df.get("name", df.get("player_name"))
    out["athlete_id"] = df.get("athlete_id")
    out["college_team"] = df.get("team", df.get("college_team"))
    out["conference"] = df.get("conference")
    out["ncaa_season"] = df.get("season", df.get("season_label"))
    out["position"] = df.get("position")
    out["games_played"] = df.get("games")
    out["games_started"] = df.get("starts")
    out["minutes"] = df.get("minutes")
    out["points"] = df.get("points")
    out["rebounds"] = _first_existing(df, ["rebounds_total", "rebounds_total_rebounds", "rebounds", "total_rebounds"])
    out["assists"] = df.get("assists")
    out["steals"] = df.get("steals")
    out["blocks"] = df.get("blocks")
    out["turnovers"] = df.get("turnovers")
    out["personal_fouls"] = df.get("fouls")
    out["usage_rate"] = df.get("usage")
    out["offensive_rating"] = df.get("offensive_rating")
    out["defensive_rating"] = df.get("defensive_rating")
    out["box_plus_minus"] = df.get("net_rating")
    out["true_shooting_percentage"] = df.get("true_shooting_pct")
    out["effective_field_goal_percentage"] = df.get("effective_field_goal_pct")
    out["assist_to_turnover_ratio"] = df.get("assists_turnover_ratio")
    out["free_throw_attempt_rate"] = df.get("free_throw_rate")
    out["offensive_rebound_rate"] = df.get("offensive_rebound_pct")

    # Nested unit stat fields from the CBBD model are flattened by pandas.json_normalize.
    out["field_goal_percentage"] = _first_existing(df, ["field_goals_percentage", "field_goals_pct", "field_goals_pct"])
    out["two_point_percentage"] = _first_existing(df, ["two_point_field_goals_percentage", "two_point_field_goals_pct"])
    out["three_point_percentage"] = _first_existing(df, ["three_point_field_goals_percentage", "three_point_field_goals_pct"])
    out["free_throw_percentage"] = _first_existing(df, ["free_throws_percentage", "free_throws_pct"])
    out["win_shares_college"] = _first_existing(df, ["win_shares_total", "win_shares", "win_shares_per_40"])

    gp = pd.to_numeric(out["games_played"], errors="coerce")
    minutes = pd.to_numeric(out["minutes"], errors="coerce")
    out["minutes_per_game"] = minutes / gp.replace(0, np.nan)
    for raw, per_game in [
        ("points", "points_per_game"), ("rebounds", "rebounds_per_game"),
        ("assists", "assists_per_game"), ("steals", "steals_per_game"),
        ("blocks", "blocks_per_game"), ("turnovers", "turnovers_per_game"),
        ("personal_fouls", "personal_fouls_per_game"),
    ]:
        out[per_game] = pd.to_numeric(out[raw], errors="coerce") / gp.replace(0, np.nan)
    return out


def _normalize_cbbd_shooting_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = _clean_api_columns(df)
    out = pd.DataFrame()
    out["player_name"] = df.get("name", df.get("player_name"))
    out["athlete_id"] = df.get("athlete_id")
    out["college_team"] = df.get("team", df.get("college_team"))
    out["ncaa_season"] = df.get("season", df.get("season_label"))
    out["three_point_attempt_rate"] = _first_existing(df, ["three_point_rate", "three_point_attempt_rate", "three_point_field_goals_rate"])
    out["free_throw_attempt_rate"] = _first_existing(df, ["free_throw_rate", "free_throw_attempt_rate"])
    out["shooting_at_rim_pct"] = _first_existing(df, ["at_rim_percentage", "at_rim_pct"])
    out["shooting_midrange_pct"] = _first_existing(df, ["mid_range_percentage", "midrange_pct"])
    return out.dropna(how="all", axis=1)


def _normalize_cbbd_team_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = _clean_api_columns(df)
    out = pd.DataFrame()
    out["college_team"] = df.get("team", df.get("school"))
    out["ncaa_season"] = df.get("season", df.get("season_label"))
    out["team_offensive_efficiency"] = _first_existing(df, ["offensive_rating", "offensive_efficiency", "offense_rating"])
    out["team_defensive_efficiency"] = _first_existing(df, ["defensive_rating", "defensive_efficiency", "defense_rating"])
    out["team_adjusted_efficiency_margin"] = _first_existing(df, ["net_rating", "adjusted_efficiency_margin", "net_efficiency"])
    out["strength_of_schedule"] = _first_existing(df, ["sos", "strength_of_schedule"])
    return out.dropna(how="all", axis=1)


def _first_existing(df: pd.DataFrame, names: list[str]) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series([np.nan] * len(df), index=df.index)


def fetch_recruiting_rankings() -> pd.DataFrame:
    recruits = _fetch_cbbd_by_year("/recruiting/players", API_RECRUITING_YEARS)
    if recruits.empty:
        return recruits
    recruits = _clean_api_columns(recruits)
    out = pd.DataFrame()
    out["player_name"] = recruits.get("name", recruits.get("player_name"))
    out["recruiting_year"] = recruits.get("year")
    out["recruiting_rank"] = _first_existing(recruits, ["ranking", "rank", "composite_rank"])
    out["recruiting_stars"] = recruits.get("stars")
    out["recruiting_rating"] = _first_existing(recruits, ["rating", "composite_rating"])
    out["college_team"] = _first_existing(recruits, ["committed_to_school", "committed_to_team", "committed_to"])
    out["position"] = recruits.get("position")
    out["recruiting_source"] = "CollegeBasketballData composite"
    return out.dropna(how="all", axis=1)


def fetch_draft_results() -> pd.DataFrame:
    """Fetch NBA draft history from nba_api and keep NCAA/college draftees."""
    try:
        from nba_api.stats.endpoints import drafthistory
    except ImportError as exc:
        raise RuntimeError("Install nba_api with `pip install nba_api` to fetch NBA draft history.") from exc

    frames = []
    for year in API_DRAFT_YEARS:
        logger.info("Fetching NBA draft history for %s", year)
        endpoint = drafthistory.DraftHistory(league_id="00", season_year_nullable=str(year))
        df = endpoint.get_data_frames()[0]
        frames.append(df)
        time.sleep(NBA_API_SLEEP_SECONDS)
    draft = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    draft = _clean_api_columns(draft)
    if draft.empty:
        return draft
    out = pd.DataFrame()
    out["nba_person_id"] = draft.get("person_id")
    out["player_name"] = draft.get("player_name")
    out["draft_year"] = pd.to_numeric(draft.get("season"), errors="coerce")
    out["draft_round"] = pd.to_numeric(draft.get("round_number"), errors="coerce")
    out["draft_pick_overall"] = pd.to_numeric(draft.get("overall_pick"), errors="coerce")
    out["draft_round_pick"] = pd.to_numeric(draft.get("round_pick"), errors="coerce")
    out["nba_team_drafted_by"] = (draft.get("team_city", "").fillna("").astype(str) + " " + draft.get("team_name", "").fillna("").astype(str)).str.strip()
    out["nba_team_abbreviation"] = draft.get("team_abbreviation")
    out["college_team"] = draft.get("organization")
    out["organization_type"] = draft.get("organization_type")
    # This project is college-player focused. Keep college/university draftees only.
    org_type = out["organization_type"].fillna("").str.lower()
    out = out[org_type.str.contains("college|university|ncaa", regex=True, na=False)].copy()
    return out


def fetch_nba_outcomes() -> pd.DataFrame:
    """Build career outcomes and first-four-season diagnostics from nba_api.

    This uses only outcome-label fields. It does not feed NBA stats into model features.
    NBA statistics are outcome labels/diagnostics only and never model inputs.
    """
    draft = load_draft_results()
    if draft.empty:
        return pd.DataFrame()
    try:
        from nba_api.stats.endpoints import playercareerstats, playerawards
    except ImportError as exc:
        raise RuntimeError("Install nba_api with `pip install nba_api` to fetch NBA outcomes.") from exc

    draft = draft[pd.to_numeric(draft["draft_year"], errors="coerce").isin(API_DRAFT_YEARS)]
    rows = []
    for _, drow in draft.iterrows():
        person_id = drow.get("nba_person_id")
        if pd.isna(person_id):
            continue
        player_name = drow.get("player_name")
        draft_year = int(drow.get("draft_year"))
        logger.info("Fetching NBA career stats for %s (%s)", player_name, person_id)
        try:
            career = playercareerstats.PlayerCareerStats(player_id=int(person_id), per_mode36="Totals")
            season_df = _clean_api_columns(career.get_data_frames()[0])
            time.sleep(NBA_API_SLEEP_SECONDS)
        except Exception as exc:
            logger.warning("NBA career stats failed for %s: %s", player_name, exc)
            continue
        if season_df.empty or "season_id" not in season_df.columns:
            continue
        season_df["season_start_year"] = pd.to_numeric(season_df["season_id"].astype(str).str[:4], errors="coerce")
        first4 = season_df[season_df["season_start_year"].between(draft_year, draft_year + 3)].copy()
        first5 = season_df[season_df["season_start_year"].between(draft_year, draft_year + 4)].copy()
        if first4.empty:
            continue
        gp = pd.to_numeric(first4.get("gp"), errors="coerce").fillna(0)
        gs = pd.to_numeric(first4.get("gs"), errors="coerce").fillna(0) if "gs" in first4 else pd.Series([0] * len(first4))
        minutes = pd.to_numeric(first4.get("min"), errors="coerce").fillna(0)
        ppg = _weighted_rate(first4, "pts", "gp")
        rpg = _weighted_rate(first4, "reb", "gp")
        apg = _weighted_rate(first4, "ast", "gp")
        peak_mpg = (minutes / gp.replace(0, np.nan)).max()
        seasons_15mpg = int(((minutes / gp.replace(0, np.nan)) >= 15).sum())
        career_by_season = season_df.assign(
            gp_numeric=pd.to_numeric(season_df.get("gp"), errors="coerce").fillna(0),
            min_numeric=pd.to_numeric(season_df.get("min"), errors="coerce").fillna(0),
        ).groupby("season_start_year", as_index=False).agg(
            gp=("gp_numeric", "max"),
            minutes=("min_numeric", "max"),
        )
        career_by_season["mpg"] = (
            career_by_season["minutes"] / career_by_season["gp"].replace(0, np.nan)
        )
        qualifying = career_by_season[
            (career_by_season["gp"] >= 40) & (career_by_season["mpg"] >= 15)
        ]
        awards = _get_career_award_indicators(person_id, playerawards)
        rows.append({
            "nba_person_id": person_id,
            "player_name": player_name,
            "draft_year": draft_year,
            "nba_games_played_first4": gp.sum(),
            "nba_games_started_first4": gs.sum(),
            "nba_minutes_first4": minutes.sum(),
            "nba_minutes_per_game_first4": minutes.sum() / gp.sum() if gp.sum() else np.nan,
            "nba_peak_minutes_per_game_first4": peak_mpg,
            "nba_points_per_game_first4": ppg,
            "nba_rebounds_per_game_first4": rpg,
            "nba_assists_per_game_first4": apg,
            "nba_win_shares_first4": np.nan,
            "nba_vorp_first4": np.nan,
            "nba_bpm_first4": np.nan,
            "nba_ws_per_48_first4": np.nan,
            "seasons_played_first4": first4["season_id"].nunique(),
            "seasons_15mpg_first4": seasons_15mpg,
            # A practical proxy: appeared in any fifth NBA season after draft.
            "second_contract_indicator": int(first5["season_start_year"].max() >= draft_year + 4) if not first5.empty else 0,
            "nba_games_played_career": career_by_season["gp"].sum(),
            "nba_minutes_career": career_by_season["minutes"].sum(),
            "seasons_played_career": career_by_season["season_start_year"].nunique(),
            "qualifying_rotation_seasons_career": len(qualifying),
            "qualifying_rotation_seasons_year5_plus": int(
                (qualifying["season_start_year"] >= draft_year + 4).sum()
            ),
            "all_star_indicator": awards["all_star_indicator"],
            "all_nba_indicator": awards["all_nba_indicator"],
            # nba_api does not expose contract values. This canonical field is
            # retained so an externally supplied contract feed can mark max
            # deals without changing the labeling methodology.
            "max_contract_indicator": 0,
        })
    return pd.DataFrame(rows)


def _weighted_rate(df: pd.DataFrame, numerator: str, denominator: str) -> float:
    if numerator not in df or denominator not in df:
        return np.nan
    num = pd.to_numeric(df[numerator], errors="coerce").fillna(0).sum()
    den = pd.to_numeric(df[denominator], errors="coerce").fillna(0).sum()
    return num / den if den else np.nan


def _get_career_award_indicators(person_id: Any, playerawards_module) -> dict[str, int]:
    try:
        awards = playerawards_module.PlayerAwards(player_id=int(person_id)).get_data_frames()[0]
        awards = _clean_api_columns(awards)
        time.sleep(0.15)
    except Exception:
        return {"all_star_indicator": 0, "all_nba_indicator": 0}
    if awards.empty:
        return {"all_star_indicator": 0, "all_nba_indicator": 0}
    text_cols = [c for c in awards.columns if any(term in c for term in ["award", "description", "name"])]
    award_text = awards[text_cols].astype(str).agg(" ".join, axis=1).str.lower() if text_cols else pd.Series([""] * len(awards))
    return {
        "all_star_indicator": int(award_text.str.contains(r"all[ -]star", regex=True).any()),
        "all_nba_indicator": int(award_text.str.contains(r"all[ -]nba", regex=True).any()),
    }


def load_historical_college_stats() -> pd.DataFrame:
    return _cache_or_fetch(RAW_FILES["college"], fetch_historical_college_stats)


def load_draft_results() -> pd.DataFrame:
    return _cache_or_fetch(RAW_FILES["draft"], fetch_draft_results)


def load_nba_outcomes() -> pd.DataFrame:
    return _cache_or_fetch(RAW_FILES["nba"], fetch_nba_outcomes)


def load_recruiting_rankings(required: bool = False) -> pd.DataFrame:
    return _cache_or_fetch(RAW_FILES["recruiting"], fetch_recruiting_rankings, required=required)


def load_espn_2026_top100(required: bool = False) -> pd.DataFrame:
    """Optional CSV only: 2026 prospects are not needed to train the model."""
    return safe_read_csv(RAW_FILES["espn_2026"], required=required)


def fetch_espn_2026_top100() -> pd.DataFrame:
    logger.warning("ESPN Top 100 is optional and is not fetched by default. Use this only for future-current-prospect predictions.")
    return pd.DataFrame()


def fetch_all_raw_data() -> dict[str, pd.DataFrame]:
    """Convenience function for an explicit data acquisition step."""
    return {
        "draft": load_draft_results(),
        "college": load_historical_college_stats(),
        "nba": load_nba_outcomes(),
        "recruiting": load_recruiting_rankings(required=False),
    }
