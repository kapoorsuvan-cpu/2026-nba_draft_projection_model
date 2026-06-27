import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from .config import MODELS_DIR, PROCESSED_FILES, REPORTS_DIR, TARGET_COLUMN, CLASS_ORDER, TEST_MIN_DRAFT_YEAR
from .train_model import evaluate_predictions
from .utils import write_csv

logger = logging.getLogger(__name__)


def _feature_names(pipe, cols):
    try:
        return pipe.named_steps["preprocess"].get_feature_names_out(cols)
    except Exception:
        return np.array(cols)


def feature_importance(artifact, df: pd.DataFrame) -> pd.DataFrame:
    pipe = artifact["pipeline"]
    cols = artifact["feature_columns"]
    model = pipe.named_steps["model"]
    names = _feature_names(pipe, cols)
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.mean(np.abs(model.coef_), axis=0)
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    return pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False)


def evaluate_best_model(df: pd.DataFrame | None = None) -> dict:
    if df is None:
        df = pd.read_csv(PROCESSED_FILES["training_labeled"])
    artifact = joblib.load(MODELS_DIR / "best_model.pkl")
    cols = artifact["feature_columns"]
    labels = list(artifact.get("classes", CLASS_ORDER))
    test = df[pd.to_numeric(df["draft_year"], errors="coerce") >= TEST_MIN_DRAFT_YEAR].copy()
    if test.empty:
        test = df.sort_values("draft_year").iloc[int(len(df)*0.8):].copy()
    X, y = test[cols], test[TARGET_COLUMN]
    proba = artifact["pipeline"].predict_proba(X)
    metrics, pred = evaluate_predictions(y, proba, labels)
    cm = pd.DataFrame(confusion_matrix(y, pred, labels=labels), index=[f"actual_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    write_csv(cm.reset_index().rename(columns={"index": "actual"}), REPORTS_DIR / "confusion_matrix_best_model.csv")
    write_csv(cm.reset_index().rename(columns={"index": "actual"}), REPORTS_DIR / "confusion_matrix.csv")

    class_report = pd.DataFrame(classification_report(y, pred, labels=labels, output_dict=True, zero_division=0)).T.reset_index().rename(columns={"index": "class"})
    write_csv(class_report, REPORTS_DIR / "best_model_classification_report.csv")

    # By position and group.
    rows_pos, rows_group = [], []
    eval_df = test.copy()
    eval_df["prediction"] = pred
    for pos, sub in eval_df.groupby("primary_position"):
        if len(sub) < 5:
            continue
        m, _ = evaluate_predictions(sub[TARGET_COLUMN], proba[eval_df.index.get_indexer(sub.index)], labels)
        m.update({"primary_position": pos, "n": len(sub)})
        rows_pos.append(m)
    for group, sub in eval_df.groupby("position_group"):
        if len(sub) < 5:
            continue
        m, _ = evaluate_predictions(sub[TARGET_COLUMN], proba[eval_df.index.get_indexer(sub.index)], labels)
        m.update({"position_group": group, "n": len(sub)})
        rows_group.append(m)
    write_csv(pd.DataFrame(rows_pos), REPORTS_DIR / "model_comparison_by_position.csv")
    write_csv(pd.DataFrame(rows_group), REPORTS_DIR / "model_comparison_by_position_group.csv")

    imp = feature_importance(artifact, df)
    write_csv(imp, REPORTS_DIR / "feature_importance_best_model.csv")
    write_csv(imp, REPORTS_DIR / "feature_importance.csv")
    # Simple position importance proxy: repeat global importance by position. True SHAP/per-position can be extended when shap is installed.
    if not imp.empty and "primary_position" in df.columns:
        pfi = pd.concat([imp.assign(primary_position=pos) for pos in sorted(df["primary_position"].dropna().unique())], ignore_index=True)
    else:
        pfi = pd.DataFrame()
    write_csv(pfi, REPORTS_DIR / "position_feature_importance_best_model.csv")
    write_csv(pfi, REPORTS_DIR / "position_feature_importance.csv")
    return metrics

if __name__ == "__main__":
    evaluate_best_model()
