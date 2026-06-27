import logging
import numpy as np
import pandas as pd
from .config import INTERIM_FILES
from .data_sources import load_nba_outcomes
from .utils import normalize_name, write_csv, coalesce_columns

logger = logging.getLogger(__name__)


def clean_nba_data(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Clean either already-aggregated first-four-year NBA outcomes or season-level rows.

    If season-level rows include nba_season_index or years_since_draft, only first four
    seasons are aggregated. Otherwise the function assumes the raw file is already at
    one row per player/draft year.
    """
    if df is None:
        df = load_nba_outcomes()
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    coalesce_columns(df, ["player_name", "player", "name"], "player_name")
    df["normalized_player_name"] = df["player_name"].map(normalize_name)
    if "draft_year" in df.columns:
        df["draft_year"] = pd.to_numeric(df["draft_year"], errors="coerce")

    season_col = "nba_season_index" if "nba_season_index" in df.columns else "years_since_draft" if "years_since_draft" in df.columns else None
    if season_col:
        first4 = df[pd.to_numeric(df[season_col], errors="coerce").between(1, 4)].copy()
        group_cols = ["normalized_player_name"] + (["draft_year"] if "draft_year" in first4.columns else [])
        agg = first4.groupby(group_cols).agg(
            player_name=("player_name", "first"),
            nba_games_played_first4=("games_played", "sum"),
            nba_games_started_first4=("games_started", "sum"),
            nba_minutes_first4=("minutes", "sum"),
            nba_peak_minutes_per_game_first4=("minutes_per_game", "max"),
            nba_points_per_game_first4=("points_per_game", "mean"),
            nba_rebounds_per_game_first4=("rebounds_per_game", "mean"),
            nba_assists_per_game_first4=("assists_per_game", "mean"),
            seasons_played_first4=(season_col, "nunique"),
        ).reset_index()
        for src, dst in [("win_shares", "nba_win_shares_first4"), ("vorp", "nba_vorp_first4"), ("bpm", "nba_bpm_first4"), ("ws_per_48", "nba_ws_per_48_first4")]:
            if src in first4.columns:
                val = first4.groupby(group_cols)[src].sum().reset_index(name=dst)
                agg = agg.merge(val, on=group_cols, how="left")
        for flag in ["second_contract_indicator", "all_star_indicator", "all_nba_indicator"]:
            if flag in first4.columns:
                val = first4.groupby(group_cols)[flag].max().reset_index(name=flag)
                agg = agg.merge(val, on=group_cols, how="left")
        out = agg
    else:
        out = df.copy()
        aliases = {
            "nba_games_played_first4": ["nba_games_played_first4", "games_played_first4", "gp_first4"],
            "nba_games_started_first4": ["nba_games_started_first4", "games_started_first4", "gs_first4"],
            "nba_minutes_first4": ["nba_minutes_first4", "minutes_first4", "mp_first4"],
            "nba_peak_minutes_per_game_first4": ["nba_peak_minutes_per_game_first4", "peak_mpg_first4"],
            "nba_win_shares_first4": ["nba_win_shares_first4", "win_shares_first4", "ws_first4"],
            "nba_vorp_first4": ["nba_vorp_first4", "vorp_first4"],
            "nba_bpm_first4": ["nba_bpm_first4", "bpm_first4"],
        }
        for target, candidates in aliases.items():
            coalesce_columns(out, candidates, target)

    for col in ["second_contract_indicator", "all_star_indicator", "all_nba_indicator"]:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    out["nba_minutes_per_game_first4"] = out.get("nba_minutes_per_game_first4", out["nba_minutes_first4"] / out.get("nba_games_played_first4", np.nan))
    write_csv(out, INTERIM_FILES["nba_clean"])
    logger.info("Saved NBA outcomes: %s rows", len(out))
    return out

if __name__ == "__main__":
    clean_nba_data()
