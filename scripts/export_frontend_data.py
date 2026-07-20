import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import joblib


OUT = Path(os.getenv("DASHBOARD_OUT", "frontend/public/data"))
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


def build_model_input(existing_rows, feature_cols):
    X = existing_rows.reindex(columns=feature_cols).copy()
    def numeric_series(name):
        return pd.to_numeric(
            existing_rows.get(name, pd.Series(np.nan, index=existing_rows.index)),
            errors="coerce",
        )

    games = numeric_series("games_played")
    mpg = numeric_series("minutes_per_game")
    per_40 = {}
    for stem in ["points", "rebounds", "assists", "steals", "blocks"]:
        per_game = numeric_series(f"{stem}_per_game")
        per_40[stem] = per_game * 40 / mpg.replace(0, np.nan)
        if stem in X:
            X[stem] = per_game * games
        if f"{stem}_per_40" in X:
            X[f"{stem}_per_40"] = per_40[stem]
    if "defensive_event_score" in X:
        X["defensive_event_score"] = 0.55 * per_40["steals"] + 0.45 * per_40["blocks"]
    if "stock_rate_per_40" in X:
        X["stock_rate_per_40"] = per_40["steals"] + per_40["blocks"]
    if "log_recruiting_rank" in X:
        X["log_recruiting_rank"] = np.log1p(numeric_series("recruiting_rank"))
    if "is_high_major" in X:
        high_major = {"ACC", "Big Ten", "Big 12", "Big East", "SEC", "Pac-12"}
        X["is_high_major"] = existing_rows.get(
            "conference", pd.Series(index=existing_rows.index)
        ).isin(high_major).astype(int)
    return X

# The repository tracks the prior exported prospect rows but intentionally
# ignores the original report/input files. Refresh those rows with the newly
# selected model rather than leaving stale classifications in the dashboard.
if (pred.empty or inp.empty) and (OUT / "prospects.json").exists() and Path("models/best_model.pkl").exists():
    existing = pd.DataFrame(json.load(open(OUT / "prospects.json")))
    prediction_frames = []
    coverage_values = pd.Series(index=existing.index, dtype=float)
    for position, group_rows in existing.groupby(existing.get("position", "").astype(str).str.upper()):
        model_path = Path(f"models/best_model_{position.lower()}.pkl")
        if not model_path.exists():
            model_path = Path("models/best_model.pkl")
        artifact = joblib.load(model_path)
        X = build_model_input(group_rows, artifact["feature_columns"])
        probabilities = artifact["pipeline"].predict_proba(X)
        classes = list(artifact["classes"])
        group_pred = pd.DataFrame({
            "player_name": group_rows["player_name"],
            "predicted_label": [classes[i] for i in probabilities.argmax(axis=1)],
            "confidence": probabilities.max(axis=1),
            "model_algorithm": artifact["algorithm"],
        })
        for label, output_col in [
            ("Star", "prob_star"),
            ("Rotation", "prob_rotation"),
            ("Not NBA Level", "prob_not_nba_level"),
        ]:
            group_pred[output_col] = (
                probabilities[:, classes.index(label)] if label in classes else 0.0
            )
        metric = best[best.get("position", pd.Series(dtype=str)).astype(str).eq(position)]
        group_pred["model_macro_f1"] = metric.iloc[0].get("macro_f1") if not metric.empty else np.nan
        group_pred["model_test_n"] = metric.iloc[0].get("test_n") if not metric.empty else np.nan
        coverage_values.loc[group_rows.index] = X.notna().mean(axis=1) * 100
        prediction_frames.append(group_pred)
    pred = pd.concat(prediction_frames, ignore_index=True)
    old_prediction_cols = [
        "predicted_label", "prob_star", "prob_rotation", "prob_not_nba_level",
        "confidence", "model_algorithm", "model_macro_f1", "model_test_n",
    ]
    inp = existing.drop(columns=old_prediction_cols, errors="ignore")
    inp["feature_coverage_pct"] = coverage_values
if best.empty:
    comparison = safe_read("reports/model_comparison_overall.csv")
    if not comparison.empty:
        selected = comparison.get("selected_model", pd.Series(False, index=comparison.index)).fillna(False)
        best = comparison[selected | comparison["algorithm"].eq("tabfm")].copy()
        best = best.rename(columns={"algorithm": "best_algorithm"})
        best["position"] = best["model_name"].apply(
            lambda name: str(name).removeprefix("position_specific_")
            if str(name).startswith("position_specific_")
            else "Overall"
        )
        best = best.drop(columns=["model_path"], errors="ignore")
fi = safe_read("reports/best_model_feature_importance_by_position.csv")
if fi.empty:
    importance_frames = []
    for model_path in sorted(Path("models").glob("best_model_[gfc].pkl")):
        artifact = joblib.load(model_path)
        pipeline = artifact.get("pipeline")
        if pipeline is None or not hasattr(pipeline, "named_steps"):
            continue
        estimator = pipeline.named_steps.get("model")
        preprocessor = pipeline.named_steps.get("preprocess")
        values = None
        if hasattr(estimator, "feature_importances_"):
            values = np.asarray(estimator.feature_importances_)
        elif hasattr(estimator, "coef_"):
            values = np.abs(np.asarray(estimator.coef_)).mean(axis=0)
            if values.max() > 0:
                values = values / values.max()
        if values is not None and preprocessor is not None:
            importance_frames.append(pd.DataFrame({
                "position": model_path.stem[-1].upper(),
                "feature": preprocessor.get_feature_names_out(),
                "importance": values,
                "algorithm": artifact.get("algorithm"),
            }))
    if importance_frames:
        fi = pd.concat(importance_frames, ignore_index=True).sort_values(
            ["position", "importance"], ascending=[True, False]
        )


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
    manifest["local_file_needed"] = manifest["slug"].map(
        lambda slug: str(OUT.parent / "headshots" / f"{slug}.jpg")
    )
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
dashboard = {
    "labelDefinitions": {
        "Star": "All-Star selection, All-NBA selection, or max contract",
        "Rotation": "At least two 40-game, 15-MPG seasons, including one in NBA year five or later",
        "Not NBA Level": "Met neither the Star nor Rotation benchmark",
    }
}

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
