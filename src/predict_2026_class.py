import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from .config import MODELS_DIR, PROCESSED_FILES
from .build_2026_prospect_dataset import build_2026_prospect_dataset
from .utils import write_csv

logger = logging.getLogger(__name__)


def _nearest_comps(prospect_row, hist: pd.DataFrame, feature_cols: list[str], k: int = 5) -> list[str]:
    subset = hist.copy()
    if "primary_position" in hist.columns and pd.notna(prospect_row.get("primary_position")):
        pos_match = hist["primary_position"] == prospect_row.get("primary_position")
        if pos_match.sum() >= k:
            subset = hist[pos_match].copy()
    # Numeric profile features for comps.
    comp_cols = [c for c in ["age_final_college_season", "height", "weight", "recruiting_rank", "usage_rate", "true_shooting_percentage", "box_plus_minus", "points_per_40", "rebounds_per_40", "assists_per_40", "stock_rate_per_40", "draft_pick_overall"] if c in subset.columns and c in prospect_row.index]
    if not comp_cols or subset.empty:
        return []
    X = subset[comp_cols].apply(pd.to_numeric, errors="coerce")
    x = pd.DataFrame([prospect_row[comp_cols]]).apply(pd.to_numeric, errors="coerce")
    med = X.median(numeric_only=True)
    X = X.fillna(med)
    x = x.fillna(med)
    scale = X.std(numeric_only=True).replace(0, 1).fillna(1)
    Xs, xs = (X - med) / scale, (x - med) / scale
    d = pairwise_distances(xs, Xs)[0]
    idx = np.argsort(d)[:k]
    return subset.iloc[idx]["player_name"].fillna("").astype(str).tolist()


def _key_features(row, cols):
    numeric = row[[c for c in cols if c in row.index]].apply(pd.to_numeric, errors="coerce").dropna()
    if numeric.empty:
        return "", ""
    positive = numeric.sort_values(ascending=False).head(5).index.tolist()
    negative = numeric.sort_values(ascending=True).head(5).index.tolist()
    return ", ".join(positive), ", ".join(negative)


def predict_2026_class(prospects: pd.DataFrame | None = None) -> pd.DataFrame:
    if prospects is None:
        if PROCESSED_FILES["prospect_features"].exists():
            prospects = pd.read_csv(PROCESSED_FILES["prospect_features"])
        else:
            prospects = build_2026_prospect_dataset()
    artifact = joblib.load(MODELS_DIR / "best_model.pkl")
    cols = artifact["feature_columns"]
    for c in cols:
        if c not in prospects.columns:
            prospects[c] = np.nan
    proba = artifact["pipeline"].predict_proba(prospects[cols])
    classes = list(artifact.get("classes", artifact["pipeline"].classes_))
    pred_idx = np.argmax(proba, axis=1)
    out = prospects.copy()
    out["predicted_success_label"] = [classes[i] for i in pred_idx]
    for label in classes:
        clean = label.lower().replace(" ", "_")
        out[f"{clean}_probability"] = proba[:, classes.index(label)]
    # Required canonical probability names.
    out["star_probability"] = out.get("star_probability", 0)
    out["rotation_probability"] = out.get("rotation_probability", 0)
    out["not_nba_level_probability"] = out.get("not_nba_level_probability", out.get("not_nba_level_probability", 0))
    out["model_confidence"] = proba.max(axis=1)
    out["nba_success_score_projection"] = out.get("star_probability", 0) * 1.0 + out.get("rotation_probability", 0) * 0.55

    hist = pd.read_csv(PROCESSED_FILES["training_labeled"])
    comps = out.apply(lambda r: _nearest_comps(r, hist, cols, 5), axis=1)
    for i in range(5):
        out[f"comp_{i+1}"] = comps.apply(lambda x: x[i] if len(x) > i else "")
    keys = out.apply(lambda r: _key_features(r, cols), axis=1)
    out["key_positive_features"] = [k[0] for k in keys]
    out["key_negative_features"] = [k[1] for k in keys]

    wanted = [
        "espn_rank", "player_name", "school_or_team", "position", "primary_position", "position_group",
        "predicted_success_label", "star_probability", "rotation_probability", "not_nba_level_probability",
        "nba_success_score_projection", "model_confidence", "key_positive_features", "key_negative_features",
        "comp_1", "comp_2", "comp_3", "comp_4", "comp_5", "data_quality_flag"
    ]
    keep = [c for c in wanted if c in out.columns] + [c for c in out.columns if c not in wanted]
    out = out[keep]
    write_csv(out, PROCESSED_FILES["prospect_predictions"])
    write_csv(out[[c for c in wanted if c in out.columns]], __import__('pathlib').Path(__file__).resolve().parents[1] / "reports" / "2026_projection_summary.csv")
    return out

if __name__ == "__main__":
    predict_2026_class()
