import numpy as np
import pandas as pd
from .config import LABEL_THRESHOLDS


def _pct_rank(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).rank(pct=True)


def create_success_labels(df: pd.DataFrame, thresholds: dict | None = None) -> pd.DataFrame:
    thresholds = thresholds or LABEL_THRESHOLDS
    out = df.copy()
    for col in [
        "nba_minutes_first4", "nba_peak_minutes_per_game_first4", "nba_win_shares_first4",
        "nba_vorp_first4", "nba_bpm_first4", "nba_games_started_first4", "second_contract_indicator",
        "all_star_indicator", "all_nba_indicator", "max_contract_indicator",
        "seasons_played_first4",
    ]:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    # Outcome classes are deliberately definition-based, not percentile-based.
    # Star: earned an All-Star or All-NBA selection, or received a max contract.
    star = (
        (out["all_star_indicator"] == 1)
        | (out["all_nba_indicator"] == 1)
        | (out["max_contract_indicator"] == 1)
    )
    # Rotation: sustained an NBA roster presence for at least three of the first
    # four seasons, without meeting the Star definition.
    rotation = (~star) & (
        out["seasons_played_first4"] >= thresholds["rotation_seasons_played_first4"]
    )

    out["nba_success_label"] = np.select([star, rotation], ["Star", "Rotation"], default="Not NBA Level")
    out["is_star"] = (out["nba_success_label"] == "Star").astype(int)
    out["is_rotation_or_better"] = out["nba_success_label"].isin(["Star", "Rotation"]).astype(int)
    out["is_not_nba_level"] = (out["nba_success_label"] == "Not NBA Level").astype(int)

    metric = out["nba_win_shares_first4"]
    if metric.abs().sum() == 0:
        metric = out["nba_vorp_first4"]
    if metric.abs().sum() == 0:
        metric = out["nba_games_started_first4"]
    out["nba_success_score"] = (
        0.35 * _pct_rank(out["nba_minutes_first4"]) +
        0.20 * _pct_rank(out["nba_peak_minutes_per_game_first4"]) +
        0.25 * _pct_rank(metric) +
        0.05 * out["all_star_indicator"] +
        0.025 * out["all_nba_indicator"] +
        0.025 * out["max_contract_indicator"]
    )
    return out
