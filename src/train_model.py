import logging
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    MODELS_DIR,
    REPORTS_DIR,
    PROCESSED_FILES,
    TARGET_COLUMN,
    TRAIN_MAX_DRAFT_YEAR,
    TEST_MIN_DRAFT_YEAR,
    TEST_MAX_DRAFT_YEAR,
    RANDOM_STATE,
    MIN_POSITION_MODEL_SAMPLE,
)
from .features import get_feature_columns
from .utils import write_csv

logger = logging.getLogger(__name__)


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )


def candidate_algorithms() -> dict:
    """
    Candidate model suite.

    XGBoost is intentionally skipped for now because it requires numeric-encoded
    target labels. We can add it back later with LabelEncoder.
    """
    algos = {
        "logistic_regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
    }

    try:
        from lightgbm import LGBMClassifier

        algos["lightgbm"] = LGBMClassifier(
            n_estimators=350,
            learning_rate=0.04,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            verbose=-1,
        )
    except Exception:
        logger.info("LightGBM not available or not compatible; skipping it.")

    return algos


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    all_cols = get_feature_columns(df, include_draft_pick=True, include_position=True)
    college_only = get_feature_columns(df, include_draft_pick=False, include_position=False)

    draft_only = [c for c in ["draft_pick_overall", "draft_round"] if c in df.columns]

    college_age = [c for c in college_only if c != "recruiting_rank"]
    college_age += [c for c in ["age_final_college_season"] if c in df.columns]

    college_recruiting = college_only + [
        c
        for c in ["recruiting_rank", "recruiting_stars", "log_recruiting_rank"]
        if c in df.columns
    ]

    college_age_recruiting = sorted(set(college_age + college_recruiting))

    position_aware = all_cols + [
        c for c in ["position", "primary_position", "position_group"] if c in df.columns
    ]

    return {
        "draft_pick_only": draft_only,
        "college_only": sorted(set(college_only)),
        "college_age": sorted(set(college_age)),
        "college_recruiting": sorted(set(college_recruiting)),
        "college_age_recruiting": sorted(set(college_age_recruiting)),
        "full": sorted(set(all_cols)),
        "position_aware": sorted(set(position_aware)),
    }


def split_by_draft_year(df: pd.DataFrame):
    draft_year = pd.to_numeric(df["draft_year"], errors="coerce")

    train = df[draft_year <= TRAIN_MAX_DRAFT_YEAR].copy()
    test = df[
        draft_year.between(TEST_MIN_DRAFT_YEAR, TEST_MAX_DRAFT_YEAR, inclusive="both")
    ].copy()

    if train.empty or test.empty:
        logger.warning(
            "Draft-year split produced empty train/test. Falling back to chronological 80/20 split."
        )
        df = df.sort_values("draft_year")
        cut = int(len(df) * 0.8)
        train = df.iloc[:cut].copy()
        test = df.iloc[cut:].copy()

    return train, test


def evaluate_predictions(y_true, proba, labels):
    labels = list(labels)
    pred = np.array(labels)[np.argmax(proba, axis=1)]

    metrics = {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, pred, average="weighted", zero_division=0),
    }

    try:
        metrics["log_loss"] = log_loss(y_true, proba, labels=labels)
    except Exception:
        metrics["log_loss"] = np.nan

    try:
        y_bin = pd.get_dummies(pd.Categorical(y_true, categories=labels)).values
        metrics["ovr_roc_auc"] = roc_auc_score(
            y_bin,
            proba,
            average="macro",
            multi_class="ovr",
        )
    except Exception:
        metrics["ovr_roc_auc"] = np.nan

    briers = []
    for i, label in enumerate(labels):
        try:
            briers.append(
                brier_score_loss((np.array(y_true) == label).astype(int), proba[:, i])
            )
        except Exception:
            pass

    metrics["brier_score"] = float(np.mean(briers)) if briers else np.nan

    return metrics, pred


def train_one_model(train, test, cols, algorithm_name, estimator, model_name):
    cols = [c for c in cols if c in train.columns]

    if not cols:
        raise ValueError(f"No usable columns for model {model_name} / {algorithm_name}")

    X_train = train[cols].copy()
    y_train = train[TARGET_COLUMN].copy()
    X_test = test[cols].copy()
    y_test = test[TARGET_COLUMN].copy()

    pipe = Pipeline(
        steps=[
            ("preprocess", make_preprocessor(X_train)),
            ("model", clone(estimator)),
        ]
    )

    pipe.fit(X_train, y_train)

    labels = list(pipe.named_steps["model"].classes_)

    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X_test)
    else:
        pred = pipe.predict(X_test)
        proba = pd.get_dummies(pd.Categorical(pred, categories=labels)).values

    metrics, pred = evaluate_predictions(y_test, proba, labels)

    metrics.update(
        {
            "model_name": model_name,
            "algorithm": algorithm_name,
            "feature_set": model_name,
            "n_features": len(cols),
            "train_n": len(train),
            "test_n": len(test),
        }
    )

    model_path = MODELS_DIR / f"{model_name}__{algorithm_name}.pkl"

    joblib.dump(
        {
            "pipeline": pipe,
            "feature_columns": cols,
            "classes": labels,
            "model_name": model_name,
            "algorithm": algorithm_name,
        },
        model_path,
    )

    metrics["model_path"] = str(model_path)

    return metrics


def train_tabfm_candidates(train: pd.DataFrame, test: pd.DataFrame, sets: dict[str, list[str]]):
    """Evaluate Google TabFM on the same global feature sets and holdout.

    TabFM performs its own mixed-type encoding and in-context inference, so the
    raw feature frame is passed directly. Only the best TabFM feature-set
    artifact is persisted to avoid serializing the foundation weights once per
    feature set.
    """
    try:
        from .tabfm_adapter import LazyTabFMClassifier, load_tabfm_model
    except Exception as exc:
        logger.info("TabFM not available; skipping it: %s", exc)
        return []

    try:
        foundation_model = load_tabfm_model()
    except Exception as exc:
        logger.warning("TabFM weights could not be loaded; skipping it: %s", exc)
        return []

    candidates = []
    # Benchmark the foundation model on the canonical full feature set. The
    # conventional suite still explores all existing ablations.
    tabfm_sets = {"full": sets["full"]} if sets.get("full") else sets
    for fs_name, cols in tabfm_sets.items():
        cols = [c for c in cols if c in train.columns]
        if not cols:
            continue
        try:
            classifier = LazyTabFMClassifier(
                model=foundation_model,
                batch_size=2,
                random_state=RANDOM_STATE,
                verbose=False,
            )
            X_train = train[cols].copy()
            X_test = test[cols].copy()
            classifier.fit(X_train, train[TARGET_COLUMN])
            proba = classifier.predict_proba(X_test)
            labels = list(classifier.classes_)
            metrics, _ = evaluate_predictions(test[TARGET_COLUMN], proba, labels)
            metrics.update(
                {
                    "model_name": fs_name,
                    "algorithm": "tabfm",
                    "feature_set": fs_name,
                    "n_features": len(cols),
                    "train_n": len(train),
                    "test_n": len(test),
                }
            )
            candidates.append(
                (
                    metrics,
                    {
                        "pipeline": classifier,
                        "feature_columns": cols,
                        "classes": labels,
                        "model_name": fs_name,
                        "algorithm": "tabfm",
                    },
                )
            )
        except Exception as exc:
            logger.warning("TabFM failed: feature_set=%s error=%s", fs_name, exc)

    if not candidates:
        return []

    best_metrics, best_artifact = max(
        candidates,
        key=lambda item: (
            item[0].get("accuracy", float("-inf")),
            item[0].get("macro_f1", float("-inf")),
            item[0].get("balanced_accuracy", float("-inf")),
        ),
    )
    model_path = MODELS_DIR / "tabfm_best.pkl"
    joblib.dump(best_artifact, model_path)
    best_metrics["model_path"] = str(model_path)
    return [best_metrics]


def train_models(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = pd.read_csv(PROCESSED_FILES["training_labeled"])

    df = df[df[TARGET_COLUMN].notna()].copy()

    train, test = split_by_draft_year(df)

    rows = []
    algos = candidate_algorithms()

    sets = feature_sets(df)
    for fs_name, cols in sets.items():
        if not cols:
            continue

        for algo_name, est in algos.items():
            try:
                rows.append(train_one_model(train, test, cols, algo_name, est, fs_name))
            except Exception as e:
                logger.warning(
                    "Model failed: feature_set=%s algorithm=%s error=%s",
                    fs_name,
                    algo_name,
                    e,
                )

    # TabFM is compared using the identical chronological split and global
    # feature-set methodology. Position subsets are intentionally not used for
    # model selection because their holdout sample sizes are not comparable.
    rows.extend(train_tabfm_candidates(train, test, sets))

    # Position-specific models where sample size allows.
    for pos in sorted(df.get("primary_position", pd.Series(dtype=str)).dropna().unique()):
        sub = df[df["primary_position"] == pos].copy()

        if len(sub) < MIN_POSITION_MODEL_SAMPLE:
            continue

        tr, te = split_by_draft_year(sub)

        if tr.empty or te.empty:
            continue

        cols = feature_sets(sub)["college_age_recruiting"]

        for algo_name, est in algos.items():
            try:
                rows.append(
                    train_one_model(
                        tr,
                        te,
                        cols,
                        algo_name,
                        est,
                        f"position_specific_{pos}",
                    )
                )
            except Exception as e:
                logger.warning(
                    "Position model failed: position=%s algorithm=%s error=%s",
                    pos,
                    algo_name,
                    e,
                )

    # Broad position-group models where sample size allows.
    for group in sorted(df.get("position_group", pd.Series(dtype=str)).dropna().unique()):
        sub = df[df["position_group"] == group].copy()

        if len(sub) < MIN_POSITION_MODEL_SAMPLE:
            continue

        tr, te = split_by_draft_year(sub)

        if tr.empty or te.empty:
            continue

        cols = feature_sets(sub)["college_age_recruiting"]

        for algo_name, est in algos.items():
            try:
                rows.append(
                    train_one_model(
                        tr,
                        te,
                        cols,
                        algo_name,
                        est,
                        f"position_group_{group}",
                    )
                )
            except Exception as e:
                logger.warning(
                    "Position-group model failed: group=%s algorithm=%s error=%s",
                    group,
                    algo_name,
                    e,
                )

    metrics = pd.DataFrame(rows)

    if metrics.empty:
        raise RuntimeError("No models trained successfully.")

    write_csv(metrics, REPORTS_DIR / "model_metrics.csv")
    write_csv(metrics, REPORTS_DIR / "model_comparison_overall.csv")

    # Convenience copies expected by the project spec.
    for src_name, out_name in [
        ("draft_pick_only", "baseline_draft_pick_model.pkl"),
        ("college_only", "college_only_model.pkl"),
        ("full", "full_model.pkl"),
    ]:
        subset = metrics[metrics["model_name"] == src_name].sort_values(
            ["macro_f1", "balanced_accuracy"],
            ascending=False,
        )

        if not subset.empty:
            shutil.copyfile(subset.iloc[0]["model_path"], MODELS_DIR / out_name)

    return metrics


if __name__ == "__main__":
    train_models()
