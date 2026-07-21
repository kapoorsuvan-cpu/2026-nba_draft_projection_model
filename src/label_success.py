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
        "seasons_played_first4", "nba_games_played_career", "nba_minutes_career",
        "seasons_played_career", "qualifying_rotation_seasons_career",
        "qualifying_rotation_seasons_year5_plus",
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
    # Rotation requires both repeated meaningful usage and market validation:
    # at least one qualifying season must occur after the rookie-contract window.
    rotation = (~star) & (
        (out["qualifying_rotation_seasons_career"] >= thresholds["rotation_min_qualifying_seasons"])
        & (
            out["qualifying_rotation_seasons_year5_plus"]
            >= thresholds["rotation_min_post_rookie_qualifying_seasons"]
        )
    )

    out["nba_success_label"] = np.select([star, rotation], ["Star", "Rotation"], default="Not NBA Level")
    out["is_star"] = (out["nba_success_label"] == "Star").astype(int)
    out["is_rotation_or_better"] = out["nba_success_label"].isin(["Star", "Rotation"]).astype(int)
    out["is_not_nba_level"] = (out["nba_success_label"] == "Not NBA Level").astype(int)

    out["nba_success_score"] = (
        0.30 * _pct_rank(out["nba_minutes_career"]) +
        0.20 * _pct_rank(out["nba_games_played_career"]) +
        0.30 * _pct_rank(out["qualifying_rotation_seasons_career"]) +
        0.05 * out["all_star_indicator"] +
        0.075 * out["all_nba_indicator"] +
        0.075 * out["max_contract_indicator"]
    )
    return out
