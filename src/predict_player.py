import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import REPORTS_DIR


REQUIRED_DISPLAY_COLS = [
    "player_name",
    "position",
]


def normalize_position(pos: str) -> str:
    """
    Convert input position to the model's supported position groups: G/F/C.
    """
    if pd.isna(pos):
        return ""

    p = str(pos).upper().strip()

    if p in {"G", "PG", "SG", "GUARD"}:
        return "G"

    if p in {"F", "SF", "PF", "FORWARD"}:
        return "F"

    if p in {"C", "CENTER", "CENTRE"}:
        return "C"

    return p


def load_best_position_models() -> dict:
    """
    Load best model metadata for each position group from report output.
    """
    best_path = REPORTS_DIR / "best_model_by_position.csv"

    if not best_path.exists():
        raise FileNotFoundError(
            "Missing reports/best_model_by_position.csv. Run pipeline and position reports first."
        )

    best = pd.read_csv(best_path)

    models = {}

    for _, row in best.iterrows():
        position = str(row.get("position", "")).upper().strip()
        model_path = row.get("model_path")

        if not position or pd.isna(model_path):
            continue

        model_path = Path(model_path)

        if not model_path.exists():
            continue

        models[position] = {
            "position": position,
            "algorithm": row.get("best_algorithm"),
            "macro_f1": row.get("macro_f1"),
            "balanced_accuracy": row.get("balanced_accuracy"),
            "accuracy": row.get("accuracy"),
            "test_n": row.get("test_n"),
            "model_path": model_path,
            "model_obj": joblib.load(model_path),
        }

    return models


def prepare_input_row(row: pd.Series, feature_columns: list[str]) -> pd.DataFrame:
    """
    Build a one-row dataframe with exactly the columns the trained model expects.
    Missing features are filled with NaN.
    """
    out = {}

    for col in feature_columns:
        out[col] = row[col] if col in row.index else np.nan

    return pd.DataFrame([out], columns=feature_columns)


def predict_one(row: pd.Series, models: dict) -> dict:
    raw_position = row.get("position", row.get("primary_position", ""))
    position = normalize_position(raw_position)

    if position not in models:
        return {
            "player_name": row.get("player_name", ""),
            "input_position": raw_position,
            "model_position": position,
            "prediction_status": "no_model_for_position",
            "predicted_label": None,
        }

    model_info = models[position]
    model_obj = model_info["model_obj"]

    pipe = model_obj["pipeline"]
    feature_columns = model_obj["feature_columns"]
    classes = list(model_obj["classes"])

    X = prepare_input_row(row, feature_columns)

    pred = pipe.predict(X)[0]

    result = {
        "player_name": row.get("player_name", ""),
        "input_position": raw_position,
        "model_position": position,
        "prediction_status": "ok",
        "predicted_label": pred,
        "model_algorithm": model_info["algorithm"],
        "model_macro_f1": model_info["macro_f1"],
        "model_balanced_accuracy": model_info["balanced_accuracy"],
        "model_accuracy": model_info["accuracy"],
        "model_test_n": model_info["test_n"],
    }

    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)[0]

        for cls, p in zip(classes, proba):
            safe_cls = str(cls).lower().replace(" ", "_").replace("-", "_")
            result[f"prob_{safe_cls}"] = float(p)

        result["confidence"] = float(np.max(proba))

    else:
        result["confidence"] = np.nan

    return result


def predict_from_csv(input_csv: str, output_csv: str):
    models = load_best_position_models()

    if not models:
        raise RuntimeError("No saved position models found.")

    df = pd.read_csv(input_csv)

    if "position" not in df.columns:
        raise ValueError("Input CSV must include a position column with G/F/C or PG/SG/SF/PF/C.")

    rows = []

    for _, row in df.iterrows():
        rows.append(predict_one(row, models))

    out = pd.DataFrame(rows)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    return out


def create_template(output_csv: str):
    """
    Create a starter CSV template with the core fields your model is likely to use.
    The prediction script can handle missing features, but more filled-in fields = better predictions.
    """
    cols = [
        "player_name",
        "position",

        # College production
        "games_played",
        "games_started",
        "minutes",
        "minutes_per_game",
        "points",
        "points_per_game",
        "points_per_40",
        "rebounds",
        "rebounds_per_game",
        "rebounds_per_40",
        "assists",
        "assists_per_game",
        "assists_per_40",
        "steals",
        "steals_per_game",
        "steals_per_40",
        "blocks",
        "blocks_per_game",
        "blocks_per_40",
        "turnovers",
        "turnovers_per_game",
        "turnovers_per_40",
        "personal_fouls",
        "personal_fouls_per_game",
        "usage_rate",

        # Engineered college signals
        "assist_to_turnover_ratio",
        "stock_rate_per_40",
        "defensive_event_score",

        # Context / recruiting
        "college_team",
        "conference",
        "is_high_major",
        "recruiting_rank",
        "log_recruiting_rank",
        "recruiting_rating",
        "recruiting_stars",
    ]

    template = pd.DataFrame(columns=cols)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output_csv, index=False)

    print(f"Created prediction input template: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Predict NBA success label for current prospects.")
    parser.add_argument("--input", help="Input CSV with prospect features.")
    parser.add_argument("--output", default="reports/prospect_predictions.csv", help="Output predictions CSV.")
    parser.add_argument("--make-template", action="store_true", help="Create a blank prediction input template.")

    args = parser.parse_args()

    if args.make_template:
        create_template(args.output)
        return

    if not args.input:
        raise ValueError("Pass --input path/to/prospects.csv or use --make-template.")

    out = predict_from_csv(args.input, args.output)

    print("\nPredictions:")
    display_cols = [
        "player_name",
        "input_position",
        "model_position",
        "predicted_label",
        "confidence",
        "model_algorithm",
        "model_macro_f1",
        "model_test_n",
    ]

    display_cols = [c for c in display_cols if c in out.columns]
    print(out[display_cols].to_string(index=False))

    print(f"\nSaved predictions to: {args.output}")


if __name__ == "__main__":
    main()
