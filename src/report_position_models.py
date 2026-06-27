import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import REPORTS_DIR, MODELS_DIR, PROCESSED_FILES, TARGET_COLUMN
from .utils import write_csv

logger = logging.getLogger(__name__)


POSITIONS = ["G", "F", "C"]


def _safe_read(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def _get_model_step(model_obj):
    pipe = model_obj["pipeline"]
    return pipe.named_steps.get("model")


def _get_preprocessor(model_obj):
    pipe = model_obj["pipeline"]
    return pipe.named_steps.get("preprocess")


def _get_feature_names(model_obj):
    """
    Get expanded model feature names after preprocessing.
    Includes one-hot encoded categorical features when available.
    """
    preprocessor = _get_preprocessor(model_obj)

    try:
        return preprocessor.get_feature_names_out()
    except Exception:
        return np.array(model_obj.get("feature_columns", []))


def _extract_feature_importance(model_obj):
    """
    Extract feature importance from tree-based models where available.
    Falls back to absolute logistic coefficients if available.
    """
    model = _get_model_step(model_obj)
    feature_names = _get_feature_names(model_obj)

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_

    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim == 2:
            values = np.mean(np.abs(coef), axis=0)
        else:
            values = np.abs(coef)

    else:
        return pd.DataFrame(columns=["feature", "importance"])

    n = min(len(feature_names), len(values))

    out = pd.DataFrame({
        "feature": feature_names[:n],
        "importance": values[:n],
    })

    out = out.sort_values("importance", ascending=False)

    return out


def best_models_by_position(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Finds the best position-specific model for each position.
    """
    rows = []

    for pos in POSITIONS:
        model_name = f"position_specific_{pos}"

        sub = metrics[metrics["model_name"] == model_name].copy()

        if sub.empty:
            rows.append({
                "position": pos,
                "status": "no_model_found",
                "best_model_name": None,
                "best_algorithm": None,
                "macro_f1": np.nan,
                "balanced_accuracy": np.nan,
                "accuracy": np.nan,
                "weighted_f1": np.nan,
                "test_n": np.nan,
                "n_features": np.nan,
                "model_path": None,
            })
            continue

        sub = sub.sort_values(
            ["macro_f1", "balanced_accuracy", "weighted_f1"],
            ascending=False,
        )

        best = sub.iloc[0]

        rows.append({
            "position": pos,
            "status": "ok",
            "best_model_name": best["model_name"],
            "best_algorithm": best["algorithm"],
            "macro_f1": best.get("macro_f1"),
            "balanced_accuracy": best.get("balanced_accuracy"),
            "accuracy": best.get("accuracy"),
            "weighted_f1": best.get("weighted_f1"),
            "log_loss": best.get("log_loss"),
            "ovr_roc_auc": best.get("ovr_roc_auc"),
            "brier_score": best.get("brier_score"),
            "test_n": best.get("test_n"),
            "train_n": best.get("train_n"),
            "n_features": best.get("n_features"),
            "model_path": best.get("model_path"),
        })

    return pd.DataFrame(rows)


def feature_importance_by_position(best_position_models: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    """
    Loads each position's best saved model and extracts top feature importances.
    """
    all_rows = []

    for _, row in best_position_models.iterrows():
        pos = row["position"]
        model_path = row.get("model_path")

        if pd.isna(model_path) or not model_path or not Path(model_path).exists():
            continue

        model_obj = joblib.load(model_path)

        imp = _extract_feature_importance(model_obj)

        if imp.empty:
            continue

        imp = imp.head(top_n).copy()
        imp.insert(0, "position", pos)
        imp.insert(1, "model_name", row.get("best_model_name"))
        imp.insert(2, "algorithm", row.get("best_algorithm"))
        imp.insert(3, "macro_f1", row.get("macro_f1"))
        imp.insert(4, "balanced_accuracy", row.get("balanced_accuracy"))

        all_rows.append(imp)

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)


def position_model_leaderboard(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Full leaderboard of all position-specific models.
    """
    rows = metrics[metrics["model_name"].astype(str).str.startswith("position_specific_")].copy()

    if rows.empty:
        return rows

    rows["position"] = rows["model_name"].str.replace("position_specific_", "", regex=False)

    rows = rows.sort_values(
        ["position", "macro_f1", "balanced_accuracy", "weighted_f1"],
        ascending=[True, False, False, False],
    )

    return rows



def position_success_feature_correlations(training: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    """
    For each position, compute correlations between PRE-DRAFT features and nba_success_score.
    Excludes NBA outcome columns to avoid leakage.
    """
    if "nba_success_score" not in training.columns:
        return pd.DataFrame()

    leakage_patterns = [
        "nba_",
        "_nba",
        "first4",
        "first_4",
        "all_star",
        "all_nba",
        "second_contract",
        "success_label",
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
        "person_id",
        "athlete_id",
        "match_scor",
        "match_score",
        "draft_year",
        "ncaa_season",
        "recruiting_year",
    ]

    direct_exclude = {
        TARGET_COLUMN,
        "nba_success_score",
        "player_name",
        "normalized_player_name",
        "is_star",
        "is_rotation_or_better",
        "is_not_nba_level",
    }

    rows = []

    for pos in POSITIONS:
        if "primary_position" not in training.columns:
            continue

        sub = training[training["primary_position"].astype(str).str.upper() == pos].copy()

        if len(sub) < 10:
            continue

        numeric_cols = sub.select_dtypes(include=[np.number]).columns.tolist()

        clean_numeric_cols = []
        for col in numeric_cols:
            lc = str(col).lower()

            if col in direct_exclude:
                continue

            if any(pattern in lc for pattern in leakage_patterns):
                continue

            clean_numeric_cols.append(col)

        for col in clean_numeric_cols:
            x = pd.to_numeric(sub[col], errors="coerce")
            y = pd.to_numeric(sub["nba_success_score"], errors="coerce")

            valid = x.notna() & y.notna()

            if valid.sum() < 10:
                continue

            # Skip constant columns.
            if x[valid].nunique() <= 1:
                continue

            corr = x[valid].corr(y[valid], method="spearman")

            if pd.isna(corr):
                continue

            rows.append({
                "position": pos,
                "feature": col,
                "spearman_corr_with_success_score": corr,
                "abs_corr": abs(corr),
                "n": int(valid.sum()),
            })

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    out = out.sort_values(["position", "abs_corr"], ascending=[True, False])

    final = []
    for pos in POSITIONS:
        sub = out[out["position"] == pos].copy()

        if sub.empty:
            continue

        positive = sub.sort_values("spearman_corr_with_success_score", ascending=False).head(top_n)
        positive["direction"] = "positive"

        negative = sub.sort_values("spearman_corr_with_success_score", ascending=True).head(top_n)
        negative["direction"] = "negative"

        final.append(pd.concat([positive, negative], ignore_index=True))

    return pd.concat(final, ignore_index=True) if final else pd.DataFrame()


def main():
    metrics = _safe_read(REPORTS_DIR / "model_comparison_overall.csv")
    training = _safe_read(PROCESSED_FILES["training_labeled"])

    leaderboard = position_model_leaderboard(metrics)
    best_by_pos = best_models_by_position(metrics)
    importance = feature_importance_by_position(best_by_pos, top_n=30)
    correlations = position_success_feature_correlations(training, top_n=25)

    write_csv(leaderboard, REPORTS_DIR / "position_model_leaderboard.csv")
    write_csv(best_by_pos, REPORTS_DIR / "best_model_by_position.csv")
    write_csv(importance, REPORTS_DIR / "best_model_feature_importance_by_position.csv")
    write_csv(correlations, REPORTS_DIR / "position_success_feature_correlations.csv")

    print("\nBest model by position:")
    print(best_by_pos[[
        "position",
        "status",
        "best_algorithm",
        "macro_f1",
        "balanced_accuracy",
        "accuracy",
        "test_n",
        "model_path",
    ]])

    print("\nSaved:")
    print("- reports/position_model_leaderboard.csv")
    print("- reports/best_model_by_position.csv")
    print("- reports/best_model_feature_importance_by_position.csv")
    print("- reports/position_success_feature_correlations.csv")


if __name__ == "__main__":
    main()
