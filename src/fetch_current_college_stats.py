import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from .config import DATA_DIR
from .utils import write_csv


CURRENT_STATS_PATH = DATA_DIR / "raw" / "current_2026_college_stats.csv"


def obj_to_dict(obj):
    if obj is None:
        return {}

    if isinstance(obj, dict):
        return obj

    if hasattr(obj, "model_dump"):
        return obj.model_dump()

    if hasattr(obj, "dict"):
        return obj.dict()

    if hasattr(obj, "to_dict"):
        return obj.to_dict()

    return {
        k: v
        for k, v in vars(obj).items()
        if not k.startswith("_")
    }


def flatten_records(records):
    rows = []

    for item in records:
        d = obj_to_dict(item)

        # Flatten one level of nested objects.
        flat = {}
        for k, v in d.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    flat[f"{k}_{kk}"] = vv
            else:
                flat[k] = v

        rows.append(flat)

    return pd.DataFrame(rows)


def fetch_player_season_stats(season: int):
    load_dotenv()

    token = os.getenv("BEARER_TOKEN") or os.getenv("CBBD_API_KEY")

    if not token:
        raise RuntimeError(
            "Missing BEARER_TOKEN or CBBD_API_KEY in .env. Add your CollegeBasketballData API token."
        )

    import cbbd

    configuration = cbbd.Configuration(
        access_token=token,
        host="https://api.collegebasketballdata.com",
    )

    with cbbd.ApiClient(configuration) as api_client:
        api = cbbd.StatsApi(api_client)

        # Different generated cbbd versions may name the parameter slightly differently.
        errors = []

        for call_style in ["season_kw", "year_kw", "positional"]:
            try:
                if call_style == "season_kw":
                    records = api.get_player_season_stats(season=season)
                elif call_style == "year_kw":
                    records = api.get_player_season_stats(year=season)
                else:
                    records = api.get_player_season_stats(season)

                return records

            except Exception as e:
                errors.append(f"{call_style}: {e}")

        raise RuntimeError(
            "Could not fetch player season stats. Tried season=, year=, and positional.\n"
            + "\n".join(errors)
        )


def standardize_current_stats(df: pd.DataFrame, season: int) -> pd.DataFrame:
    df = df.copy()

    # Normalize column names to match your cleaning pipeline style.
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Common CBBD field variations.
    rename_map = {
        "player": "player_name",
        "name": "player_name",
        "team": "college_team",
        "school": "college_team",
        "season": "ncaa_season",
        "year": "ncaa_season",
        "pos": "position",
        "games": "games_played",
        "games_started": "games_started",
        "starts": "games_started",
        "minutes": "minutes",
        "points": "points",
        "rebounds": "rebounds",
        "assists": "assists",
        "steals": "steals",
        "blocks": "blocks",
        "turnovers": "turnovers",
        "fouls": "personal_fouls",
    }

    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    if "ncaa_season" not in df.columns:
        df["ncaa_season"] = season

    # Keep season explicit.
    df["ncaa_season"] = pd.to_numeric(df["ncaa_season"], errors="coerce").fillna(season)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--season",
        type=int,
        default=2026,
        help="College basketball season year. For 2025-26 prospects, use 2026.",
    )
    parser.add_argument(
        "--output",
        default=str(CURRENT_STATS_PATH),
        help="Output CSV path.",
    )

    args = parser.parse_args()

    print(f"Fetching CollegeBasketballData player season stats for season={args.season}...")

    records = fetch_player_season_stats(args.season)
    df = flatten_records(records)
    df = standardize_current_stats(df, args.season)

    write_csv(df, Path(args.output))

    print(f"Saved current season stats: {args.output}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    if "player_name" in df.columns:
        print("\nSample players:")
        print(df[["player_name"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
