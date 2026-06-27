import json
import shutil
import joblib
import numpy as np
import pandas as pd
from .config import REPORTS_DIR, MODELS_DIR, PROCESSED_FILES, TARGET_COLUMN
from .utils import write_csv


def compare_and_select_best(metrics: pd.DataFrame | None = None) -> pd.DataFrame:
    if metrics is None:
        metrics = pd.read_csv(REPORTS_DIR / "model_metrics.csv")
    metrics = metrics.copy()
    baseline = metrics[metrics["model_name"] == "draft_pick_only"]["macro_f1"].max()
    metrics["lift_over_draft_pick_baseline"] = metrics["macro_f1"] - baseline if pd.notna(baseline) else np.nan
    # Lower is better for log loss and Brier, so invert after rank.
    metrics["selection_score"] = (
        metrics["macro_f1"].rank(pct=True) * 0.35 +
        metrics["balanced_accuracy"].rank(pct=True) * 0.25 +
        metrics["weighted_f1"].rank(pct=True) * 0.15 +
        (-metrics["brier_score"]).rank(pct=True) * 0.15 +
        metrics["lift_over_draft_pick_baseline"].fillna(0).rank(pct=True) * 0.10
    )
    ranked = metrics.sort_values("selection_score", ascending=False)
    best = ranked.iloc[0].to_dict()
    shutil.copyfile(best["model_path"], MODELS_DIR / "best_model.pkl")

    df = pd.read_csv(PROCESSED_FILES["training_labeled"])
    artifact = joblib.load(MODELS_DIR / "best_model.pkl")
    metadata = {
        "model_type": best.get("algorithm"),
        "model_name": best.get("model_name"),
        "training_years": [int(df["draft_year"].min()), int(df[df["draft_year"] <= 2019]["draft_year"].max())] if "draft_year" in df else None,
        "test_years": [int(df[df["draft_year"] >= 2020]["draft_year"].min()), int(df["draft_year"].max())] if "draft_year" in df else None,
        "feature_list": artifact.get("feature_columns", []),
        "target_label_definition": "Star / Rotation / Not NBA Level using first-four NBA season outcomes; NBA outcomes are labels only, never model inputs.",
        "class_distribution": df[TARGET_COLUMN].value_counts(dropna=False).to_dict() if TARGET_COLUMN in df else {},
        "performance_metrics": {k: best.get(k) for k in ["macro_f1", "weighted_f1", "balanced_accuracy", "accuracy", "log_loss", "brier_score", "ovr_roc_auc", "lift_over_draft_pick_baseline"]},
        "whether_draft_pick_was_included": "draft_pick_overall" in artifact.get("feature_columns", []),
        "whether_position_specific_models_were_used": str(best.get("model_name", "")).startswith("position_specific_"),
        "whether_broad_position_group_models_were_used": str(best.get("model_name", "")).startswith("position_group_"),
        "limitations": [
            "Model quality depends on completeness and consistency of raw NCAA, recruiting, draft, and NBA outcome CSVs.",
            "Draft pick can dominate predictions because it embeds NBA scouting information.",
            "Current prospect projections may be weak for international/pro pathway or low-minutes players.",
        ],
    }
    (MODELS_DIR / "best_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_csv(ranked, REPORTS_DIR / "model_comparison_overall.csv")
    write_csv(pd.DataFrame([best]), REPORTS_DIR / "best_model_summary.csv")
    return ranked

if __name__ == "__main__":
    compare_and_select_best()
