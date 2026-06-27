import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path("frontend/public/data")
OUT.mkdir(parents=True, exist_ok=True)


def slugify(name):
    s = str(name).lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def safe_read(path):
    path = Path(path)
    if not path.exists():
        print(f"Missing: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def clean_json_value(v):
    if v is None:
        return None

    if isinstance(v, float):
        if np.isnan(v) or np.isinf(v):
            return None
        return float(v)

    if isinstance(v, np.floating):
        v = float(v)
        if np.isnan(v) or np.isinf(v):
            return None
        return v

    if isinstance(v, np.integer):
        return int(v)

    if isinstance(v, dict):
        return {str(k): clean_json_value(val) for k, val in v.items()}

    if isinstance(v, list):
        return [clean_json_value(x) for x in v]

    return v


def df_records(df):
    if df.empty:
        return []

    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.where(pd.notna(df), None)

    records = df.to_dict(orient="records")
    return clean_json_value(records)


def export_json(obj, path):
    obj = clean_json_value(obj)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
    print(f"Saved {path}")


pred = safe_read("reports/espn_2026_top100_predictions.csv")
inp = safe_read("data/processed/espn_2026_top100_model_input.csv")
coverage = safe_read("reports/espn_2026_top100_feature_coverage.csv")
training = safe_read("data/processed/model_training_dataset_with_labels.csv")
best = safe_read("reports/best_model_by_position.csv")
fi = safe_read("reports/best_model_feature_importance_by_position.csv")


# -----------------------------
# Prospect-level data
# -----------------------------
if not pred.empty and not inp.empty:
    prospects = inp.merge(pred, on="player_name", how="left", suffixes=("", "_pred"))

    if not coverage.empty and "player_name" in coverage.columns:
        keep_cov = [c for c in ["player_name", "feature_coverage_pct", "missing_features"] if c in coverage.columns]
        prospects = prospects.merge(coverage[keep_cov], on="player_name", how="left")

    prospects["slug"] = prospects["player_name"].map(slugify)
    prospects["headshot_url"] = "/headshots/" + prospects["slug"] + ".jpg"

    keep = [
        "espn_rank",
        "player_name",
        "slug",
        "headshot_url",
        "position",
        "espn_position",
        "college_team",
        "conference",
        "games_played",
        "minutes_per_game",
        "points_per_game",
        "rebounds_per_game",
        "assists_per_game",
        "steals_per_game",
        "blocks_per_game",
        "usage_rate",
        "recruiting_rank",
        "recruiting_rating",
        "recruiting_stars",
        "feature_coverage_pct",
        "predicted_label",
        "prob_star",
        "prob_rotation",
        "prob_not_nba_level",
        "confidence",
        "model_algorithm",
        "model_macro_f1",
        "model_test_n",
    ]

    keep = [c for c in keep if c in prospects.columns]
    prospects = prospects[keep].sort_values("espn_rank")

    export_json(df_records(prospects), OUT / "prospects.json")

    manifest = prospects[["player_name", "slug", "headshot_url"]].copy()
    manifest["local_file_needed"] = "frontend/public" + manifest["headshot_url"]
    manifest.to_csv(OUT / "headshot_manifest.csv", index=False)
    print(f"Saved {OUT / 'headshot_manifest.csv'}")
else:
    print("Could not export prospects.json because prediction/input data is missing.")


# -----------------------------
# Model metrics
# -----------------------------
if not best.empty:
    export_json(df_records(best), OUT / "model_metrics.json")
else:
    export_json([], OUT / "model_metrics.json")


# -----------------------------
# Feature importance
# -----------------------------
if not fi.empty:
    export_json(df_records(fi), OUT / "feature_importance.json")
else:
    export_json([], OUT / "feature_importance.json")


# -----------------------------
# Dashboard summary
# -----------------------------
dashboard = {}

if not training.empty:
    label_col = "nba_success_label"
    pos_col = "primary_position" if "primary_position" in training.columns else "position"

    hist = training.copy()

    if pos_col in hist.columns:
        hist[pos_col] = hist[pos_col].replace(
            {
                "PG": "G",
                "SG": "G",
                "SF": "F",
                "PF": "F",
                "Guard": "G",
                "Forward": "F",
                "Center": "C",
            }
        )
        hist = hist[hist[pos_col].isin(["G", "F", "C"])]

    if label_col in hist.columns:
        outcome_counts = (
            hist[label_col]
            .value_counts()
            .rename_axis("outcome")
            .reset_index(name="count")
        )
        outcome_counts["share"] = outcome_counts["count"] / outcome_counts["count"].sum()
        dashboard["historicalOutcomeDistribution"] = df_records(outcome_counts)

        if pos_col in hist.columns:
            by_pos = (
                hist.groupby([pos_col, label_col])
                .size()
                .reset_index(name="count")
                .rename(columns={pos_col: "position", label_col: "outcome"})
            )
            by_pos["total"] = by_pos.groupby("position")["count"].transform("sum")
            by_pos["share"] = by_pos["count"] / by_pos["total"]
            dashboard["historicalOutcomeByPosition"] = df_records(by_pos)

    if "draft_year" in hist.columns and label_col in hist.columns:
        yearly = (
            hist.groupby(["draft_year", label_col])
            .size()
            .reset_index(name="count")
            .rename(columns={label_col: "outcome"})
        )
        dashboard["historicalOutcomeByYear"] = df_records(yearly)

    dashboard["trainingSummary"] = {
        "rows": int(len(hist)),
        "minDraftYear": int(hist["draft_year"].min()) if "draft_year" in hist.columns and len(hist) else None,
        "maxDraftYear": int(hist["draft_year"].max()) if "draft_year" in hist.columns and len(hist) else None,
    }

if not pred.empty and "predicted_label" in pred.columns:
    pred_counts = (
        pred["predicted_label"]
        .value_counts()
        .rename_axis("outcome")
        .reset_index(name="count")
    )
    pred_counts["share"] = pred_counts["count"] / pred_counts["count"].sum()
    dashboard["currentClassOutcomeDistribution"] = df_records(pred_counts)

    prob_cols = [c for c in ["prob_star", "prob_rotation", "prob_not_nba_level"] if c in pred.columns]
    dashboard["currentClassAverageProbabilities"] = [
        {
            "outcome": c.replace("prob_", "").replace("_", " ").title(),
            "probability": float(pred[c].mean()),
        }
        for c in prob_cols
    ]

export_json(dashboard, OUT / "dashboard_summary.json")


# -----------------------------
# Validate all JSON outputs
# -----------------------------
for path in [
    OUT / "prospects.json",
    OUT / "model_metrics.json",
    OUT / "feature_importance.json",
    OUT / "dashboard_summary.json",
]:
    if path.exists():
        with open(path) as f:
            json.load(f)
        print(f"Validated JSON: {path}")
