import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import DATA_DIR, REPORTS_DIR
from .clean_college_data import clean_college_data, select_final_college_seasons
from .clean_recruiting_data import clean_recruiting_data
from .utils import normalize_name, fuzzy_merge, write_csv


CURRENT_STATS_PATH = DATA_DIR / "raw" / "current_2026_college_stats.csv"
MODEL_INPUT_PATH = DATA_DIR / "processed" / "espn_2026_top100_model_input.csv"
MATCH_REPORT_PATH = REPORTS_DIR / "espn_2026_top100_match_report.csv"
FEATURE_COVERAGE_PATH = REPORTS_DIR / "espn_2026_top100_feature_coverage.csv"


HIGH_MAJOR = {
    "ACC",
    "SEC",
    "Big Ten",
    "Big 10",
    "Big 12",
    "Big East",
    "Pac-12",
    "Pac 12",
}


def normalize_position_group(value):
    if pd.isna(value):
        return ""

    p = str(value).upper().strip()

    if p in {"G", "GUARD", "PG", "SG"}:
        return "G"

    if p in {"F", "FORWARD", "SF", "PF"}:
        return "F"

    if p in {"C", "CENTER", "CENTRE"}:
        return "C"

    return p


def clean_top100(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    df = df.rename(
        columns={
            "Rank": "espn_rank",
            "Player": "player_name",
            "ESPN Position": "espn_position",
            "Position Group": "position",
        }
    )

    required = ["espn_rank", "player_name", "espn_position", "position"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Input ESPN file missing required columns: {missing}")

    df["player_name"] = df["player_name"].astype(str).str.strip()
    df["normalized_player_name"] = df["player_name"].map(normalize_name)
    df["position"] = df["position"].map(normalize_position_group)

    # Use the ESPN Top 100 board rank as the prospect/recruiting rank.
    # This is a pre-draft prospect signal, not NBA draft-pick leakage.
    df["recruiting_rank"] = pd.to_numeric(df["espn_rank"], errors="coerce")

    # Rating-like score derived from rank.
    # Rank 1 = 1.000, rank 100 = 0.901.
    df["recruiting_rating"] = 1.0 - ((df["recruiting_rank"] - 1) / 1000.0)

    # Approximate star buckets from ESPN Top 100 rank.
    # Top 25 = 5-star, 26-75 = 4-star, 76-100 = 3-star.
    df["recruiting_stars"] = pd.cut(
        df["recruiting_rank"],
        bins=[0, 25, 75, 100],
        labels=[5, 4, 3],
        include_lowest=True
    ).astype(float)

    df["recruiting_source"] = "ESPN Top 100 rank used as recruiting/prospect rank"

    keep_cols = required + [
        "normalized_player_name",
        "recruiting_rank",
        "recruiting_rating",
        "recruiting_stars",
        "recruiting_source",
    ]

    return df[keep_cols].copy()


def safe_num(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def standardize_current_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    rename_map = {
        "player": "player_name",
        "name": "player_name",
        "team": "college_team",
        "school": "college_team",
        "season": "ncaa_season",
        "year": "ncaa_season",
        "pos": "position",
        "games": "games_played",
        "starts": "games_started",
        "rebounds_total": "rebounds",
        "fouls": "personal_fouls",
        "usage": "usage_rate",
        "assists_turnover_ratio": "assist_to_turnover_ratio",
        "effective_field_goal_pct": "effective_field_goal_percentage",
        "true_shooting_pct": "true_shooting_percentage",
        "field_goals_pct": "field_goal_percentage",
        "free_throws_pct": "free_throw_percentage",
        "three_point_field_goals_pct": "three_point_percentage",
        "two_point_field_goals_pct": "two_point_percentage",
    }

    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    if "ncaa_season" not in df.columns:
        df["ncaa_season"] = 2026

    return df


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    minutes = safe_num(df, "minutes")
    games = safe_num(df, "games_played")

    for stat in ["points", "rebounds", "assists", "steals", "blocks", "turnovers", "personal_fouls"]:
        total = safe_num(df, stat)
        per_game_col = f"{stat}_per_game"

        if per_game_col not in df.columns:
            df[per_game_col] = np.where(games > 0, total / games, np.nan)

    for stat in ["points", "rebounds", "assists", "steals", "blocks", "turnovers"]:
        total = safe_num(df, stat)
        per_40_col = f"{stat}_per_40"

        if per_40_col not in df.columns:
            df[per_40_col] = np.where(minutes > 0, total / minutes * 40, np.nan)

    if "minutes_per_game" not in df.columns:
        df["minutes_per_game"] = np.where(games > 0, minutes / games, np.nan)

    if "assist_to_turnover_ratio" not in df.columns:
        ast = safe_num(df, "assists")
        tov = safe_num(df, "turnovers")
        df["assist_to_turnover_ratio"] = np.where(tov > 0, ast / tov, np.nan)

    df["stock_rate_per_40"] = safe_num(df, "steals_per_40") + safe_num(df, "blocks_per_40")
    df["defensive_event_score"] = (
        0.55 * safe_num(df, "steals_per_40") +
        0.45 * safe_num(df, "blocks_per_40")
    )

    if "conference" in df.columns:
        df["conference"] = df["conference"].astype(str)
        df["is_high_major"] = df["conference"].isin(HIGH_MAJOR).astype(int)
    else:
        df["conference"] = np.nan
        df["is_high_major"] = np.nan

    if "recruiting_rank" in df.columns:
        rank = safe_num(df, "recruiting_rank")
        df["log_recruiting_rank"] = np.log1p(rank)
    elif "log_recruiting_rank" not in df.columns:
        df["log_recruiting_rank"] = np.nan

    return df


def load_current_college_and_recruiting():
    if not CURRENT_STATS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CURRENT_STATS_PATH}. Run: python -m src.fetch_current_college_stats --season 2026 --output data/raw/current_2026_college_stats.csv"
        )

    print(f"Loading current 2025-26 stats: {CURRENT_STATS_PATH}")

    raw_college = pd.read_csv(CURRENT_STATS_PATH)
    raw_college = standardize_current_stats(raw_college)

    college_clean = clean_college_data(raw_college)

    # For current prospects, do not filter by historical draft year.
    final_college = select_final_college_seasons(college_clean, draft_df=None)

    try:
        recruiting = clean_recruiting_data()
    except Exception:
        recruiting = pd.DataFrame()

    return final_college, recruiting


def merge_top100_with_features(top100: pd.DataFrame, college: pd.DataFrame, recruiting: pd.DataFrame):
    merged, college_report = fuzzy_merge(
        top100,
        college,
        left_name_col="normalized_player_name",
        right_name_col="normalized_player_name",
        score_cutoff=82,
        suffix="_college",
    )

    if recruiting is not None and not recruiting.empty and "normalized_player_name" in recruiting.columns:
        merged, recruiting_report = fuzzy_merge(
            merged,
            recruiting,
            left_name_col="normalized_player_name",
            right_name_col="normalized_player_name",
            score_cutoff=82,
            suffix="_recruiting",
        )
    else:
        recruiting_report = pd.DataFrame()

    college_report["source"] = "college"
    recruiting_report["source"] = "recruiting"

    report = pd.concat([college_report, recruiting_report], ignore_index=True)

    return merged, report


def load_required_model_features():
    best_path = REPORTS_DIR / "best_model_by_position.csv"

    if not best_path.exists():
        raise FileNotFoundError(
            "Missing reports/best_model_by_position.csv. Run pipeline and position reports first."
        )

    best = pd.read_csv(best_path)

    needed_by_pos = {}

    for _, row in best.iterrows():
        pos = str(row.get("position", "")).upper().strip()
        model_path = row.get("model_path")

        if not pos or pd.isna(model_path):
            continue

        path = Path(model_path)

        if not path.exists():
            continue

        obj = joblib.load(path)
        needed_by_pos[pos] = list(obj.get("feature_columns", []))

    return needed_by_pos


def build_feature_coverage(df: pd.DataFrame, needed_by_pos: dict) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        pos = normalize_position_group(row.get("position"))
        needed = needed_by_pos.get(pos, [])

        present = 0
        missing = []

        for col in needed:
            if col in df.columns and pd.notna(row.get(col)):
                present += 1
            else:
                missing.append(col)

        rows.append(
            {
                "espn_rank": row.get("espn_rank"),
                "player_name": row.get("player_name"),
                "position": pos,
                "features_needed": len(needed),
                "features_present": present,
                "feature_coverage_pct": present / len(needed) if needed else np.nan,
                "missing_features": ", ".join(missing),
            }
        )

    return pd.DataFrame(rows)


def build_dataset(input_csv: str | Path, output_csv: str | Path = MODEL_INPUT_PATH):
    top100 = clean_top100(input_csv)

    college, recruiting = load_current_college_and_recruiting()

    merged, match_report = merge_top100_with_features(top100, college, recruiting)

    merged = add_model_features(merged)

    needed_by_pos = load_required_model_features()
    feature_coverage = build_feature_coverage(merged, needed_by_pos)

    all_needed = sorted(set(sum(needed_by_pos.values(), [])))

    base_cols = [
        "espn_rank",
        "player_name",
        "espn_position",
        "position",
        "college_team",
        "conference",
    ]

    final_cols = []

    for col in base_cols + all_needed:
        if col in merged.columns and col not in final_cols:
            final_cols.append(col)

    final = merged[final_cols].copy()

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    write_csv(final, output_csv)
    write_csv(match_report, MATCH_REPORT_PATH)
    write_csv(feature_coverage, FEATURE_COVERAGE_PATH)

    print(f"\nSaved model input dataset: {output_csv}")
    print(f"Saved match report: {MATCH_REPORT_PATH}")
    print(f"Saved feature coverage report: {FEATURE_COVERAGE_PATH}")

    print("\nFeature coverage summary:")
    print(
        feature_coverage.groupby("position")[["features_present", "features_needed", "feature_coverage_pct"]]
        .mean()
        .round(3)
    )

    print("\nLowest coverage players:")
    print(
        feature_coverage.sort_values("feature_coverage_pct")
        .head(15)[["espn_rank", "player_name", "position", "feature_coverage_pct", "missing_features"]]
        .to_string(index=False)
    )

    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/raw/espn_2026_top100_position_groups.csv",
        help="CSV with Rank, Player, ESPN Position, Position Group.",
    )
    parser.add_argument(
        "--output",
        default=str(MODEL_INPUT_PATH),
        help="Output CSV for prediction-ready prospect model input.",
    )

    args = parser.parse_args()

    build_dataset(args.input, args.output)


if __name__ == "__main__":
    main()

# Backward-compatible wrapper for run_pipeline.py
def build_2026_prospect_dataset():
    return build_dataset(
        input_csv="data/raw/espn_2026_top100_position_groups.csv",
        output_csv="data/processed/espn_2026_top100_model_input.csv",
    )
