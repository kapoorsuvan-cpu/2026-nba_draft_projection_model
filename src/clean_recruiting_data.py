import logging
import pandas as pd
from .config import INTERIM_FILES
from .data_sources import load_recruiting_rankings
from .utils import normalize_name, write_csv, coalesce_columns

logger = logging.getLogger(__name__)


def clean_recruiting_data(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = load_recruiting_rankings(required=False)
    if df.empty:
        out = pd.DataFrame(columns=["player_name", "normalized_player_name", "recruiting_rank", "recruiting_stars", "recruiting_source"])
        write_csv(out, INTERIM_FILES["recruiting_clean"])
        return out
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    coalesce_columns(df, ["player_name", "player", "name"], "player_name")
    coalesce_columns(df, ["rank", "recruiting_rank", "national_rank"], "recruiting_rank")
    coalesce_columns(df, ["stars", "recruiting_stars"], "recruiting_stars")
    coalesce_columns(df, ["source", "recruiting_source"], "recruiting_source", default="manual")
    df["normalized_player_name"] = df["player_name"].map(normalize_name)
    df["recruiting_rank"] = pd.to_numeric(df["recruiting_rank"], errors="coerce")
    df["recruiting_stars"] = pd.to_numeric(df["recruiting_stars"], errors="coerce")
    out = df.sort_values("recruiting_rank", na_position="last").drop_duplicates("normalized_player_name")
    write_csv(out, INTERIM_FILES["recruiting_clean"])
    logger.info("Saved recruiting data: %s rows", len(out))
    return out

if __name__ == "__main__":
    clean_recruiting_data()
