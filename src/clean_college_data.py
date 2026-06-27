import logging
import numpy as np
import pandas as pd
from .config import INTERIM_FILES
from .data_sources import load_historical_college_stats
from .utils import normalize_name, normalize_team, parse_height_to_inches, standardize_position, write_csv, coalesce_columns

logger = logging.getLogger(__name__)

COLUMN_ALIASES = {
    "player_name": ["player_name", "player", "name"],
    "college_team": ["college_team", "team", "school"],
    "conference": ["conference", "conf"],
    "ncaa_season": ["ncaa_season", "season", "year"],
    "position": ["position", "pos"],
    "height": ["height", "ht"],
    "weight": ["weight", "wt"],
    "class_year": ["class_year", "class", "yr"],
}


def clean_college_data(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = load_historical_college_stats()
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    for target, candidates in COLUMN_ALIASES.items():
        coalesce_columns(df, candidates, target)

    # Ensure expected columns exist even when an API does not provide them.
    for required_col in ["player_name", "college_team", "conference", "ncaa_season", "position", "height", "weight", "class_year"]:
        if required_col not in df.columns:
            df[required_col] = np.nan

    df["player_name"] = df["player_name"].astype(str).str.strip()
    df["normalized_player_name"] = df["player_name"].map(normalize_name)
    df["college_team"] = df["college_team"].astype(str).str.strip()
    df["normalized_college_team"] = df["college_team"].map(normalize_team)
    df["ncaa_season"] = pd.to_numeric(df["ncaa_season"], errors="coerce")
    df["height"] = df["height"].map(parse_height_to_inches)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")

    pos = df["position"].apply(standardize_position)
    df["primary_position"] = [p[0] for p in pos]
    df["secondary_position"] = [p[1] for p in pos]
    df["position_group"] = [p[2] for p in pos]

    # Create specific-position label for analysis.
    # Source data often only provides broad G/F/C. For broad G/F, infer a likely
    # PG/SG/SF/PF using pre-draft physical and college-production signals.
    def derive_analysis_position(row):
        primary = str(row.get("primary_position", "")).upper()
        raw_pos = str(row.get("position", "")).upper().replace(" ", "")

        if primary in {"PG", "SG", "SF", "PF", "C"}:
            return primary

        height = pd.to_numeric(row.get("height"), errors="coerce")
        ast_pg = pd.to_numeric(row.get("assists_per_game"), errors="coerce")
        ast_40 = pd.to_numeric(row.get("assists_per_40"), errors="coerce")
        atr = pd.to_numeric(row.get("assist_to_turnover_ratio"), errors="coerce")
        reb_pg = pd.to_numeric(row.get("rebounds_per_game"), errors="coerce")
        reb_40 = pd.to_numeric(row.get("rebounds_per_40"), errors="coerce")
        blk_pg = pd.to_numeric(row.get("blocks_per_game"), errors="coerce")
        blk_40 = pd.to_numeric(row.get("blocks_per_40"), errors="coerce")

        # Broad guard split: PG if smaller and/or meaningfully playmaking.
        if primary == "G" or raw_pos == "G":
            pg_score = 0

            if pd.notna(height) and height <= 75:
                pg_score += 1
            if pd.notna(ast_pg) and ast_pg >= 4.0:
                pg_score += 2
            if pd.notna(ast_40) and ast_40 >= 5.0:
                pg_score += 2
            if pd.notna(atr) and atr >= 1.5:
                pg_score += 1

            return "PG" if pg_score >= 2 else "SG"

        # Broad forward split: PF if bigger and/or interior production.
        if primary == "F" or raw_pos == "F":
            pf_score = 0

            if pd.notna(height) and height >= 80:
                pf_score += 1
            if pd.notna(reb_pg) and reb_pg >= 6.0:
                pf_score += 1
            if pd.notna(reb_40) and reb_40 >= 8.0:
                pf_score += 1
            if pd.notna(blk_pg) and blk_pg >= 0.8:
                pf_score += 1
            if pd.notna(blk_40) and blk_40 >= 1.0:
                pf_score += 1

            return "PF" if pf_score >= 2 else "SF"

        if primary == "C" or raw_pos == "C":
            return "C"

        return ""

    df["analysis_position"] = df.apply(derive_analysis_position, axis=1)

    for col in df.columns:
        if col not in {"player_name", "normalized_player_name", "college_team", "normalized_college_team", "conference", "position", "primary_position", "secondary_position", "position_group", "analysis_position", "class_year", "birth_date"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    write_csv(df, INTERIM_FILES["college_clean"])
    logger.info("Saved cleaned college data: %s rows", len(df))
    return df


def select_final_college_seasons(college_df: pd.DataFrame, draft_df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = college_df.copy()
    if draft_df is not None and not draft_df.empty and "draft_year" in draft_df.columns:
        # If draft data are available, keep the most recent NCAA season not after the draft year.
        draft = draft_df.copy()
        draft["normalized_player_name"] = draft.get("player_name", draft.get("name", "")).map(normalize_name)
        draft["draft_year"] = pd.to_numeric(draft["draft_year"], errors="coerce")
        df = df.merge(draft[["normalized_player_name", "draft_year"]].drop_duplicates(), on="normalized_player_name", how="left")
        df = df[(df["draft_year"].isna()) | (df["ncaa_season"] <= df["draft_year"])]
    sort_cols = ["normalized_player_name", "ncaa_season"]
    final = df.sort_values(sort_cols).groupby("normalized_player_name", as_index=False).tail(1)
    write_csv(final, INTERIM_FILES["final_seasons"])
    logger.info("Saved final college seasons: %s rows", len(final))
    return final


def main():
    clean_college_data()

if __name__ == "__main__":
    main()
