import logging
import json
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
from sklearn.model_selection import ParameterGrid
from sklearn.inspection import permutation_importance

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
    OUTCOME_COLUMNS,
    POSITION_MODEL_OVERRIDES,
)
from .features import get_feature_columns
from .analyze_correlations_by_position import analyze_subset
from .estimators import LabelEncodedClassifier
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

    try:
        from xgboost import XGBClassifier

        algos["xgboost"] = LabelEncodedClassifier(
            XGBClassifier(
                n_estimators=400,
                learning_rate=0.04,
                max_depth=4,
                min_child_weight=5,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=2.0,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        )
    except Exception:
        logger.info("XGBoost not available or not compatible; skipping it.")

    return algos


def tuning_grids() -> dict[str, list[dict]]:
    grids = {
        "logistic_regression": list(ParameterGrid({"C": [0.1, 1.0, 10.0]})),
        "random_forest": list(
            ParameterGrid(
                {
                    "n_estimators": [350, 700],
                    "min_samples_leaf": [2, 6],
                    "max_features": ["sqrt", 0.7],
                }
            )
        ),
        "hist_gradient_boosting": list(
            ParameterGrid(
                {
                    "learning_rate": [0.04, 0.08],
                    "max_leaf_nodes": [15, 31],
                    "l2_regularization": [0.0, 1.0],
                }
            )
        ),
        "lightgbm": [
            {"n_estimators": 250, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 15},
            {"n_estimators": 400, "learning_rate": 0.04, "num_leaves": 31, "min_child_samples": 15},
            {"n_estimators": 500, "learning_rate": 0.025, "num_leaves": 31, "min_child_samples": 25},
            {"n_estimators": 300, "learning_rate": 0.06, "num_leaves": 15, "min_child_samples": 25},
        ],
        "xgboost": [
            {"estimator__n_estimators": 300, "estimator__learning_rate": 0.03, "estimator__max_depth": 3, "estimator__min_child_weight": 3},
            {"estimator__n_estimators": 450, "estimator__learning_rate": 0.04, "estimator__max_depth": 4, "estimator__min_child_weight": 5},
            {"estimator__n_estimators": 600, "estimator__learning_rate": 0.025, "estimator__max_depth": 5, "estimator__min_child_weight": 5},
            {"estimator__n_estimators": 350, "estimator__learning_rate": 0.06, "estimator__max_depth": 3, "estimator__min_child_weight": 8},
        ],
    }
    return grids


def select_position_features(
    train: pd.DataFrame,
    candidate_cols: list[str],
    position: str,
    max_numeric_features: int = 18,
) -> tuple[list[str], pd.DataFrame]:
    """Select non-redundant position features from train-only association reports."""
    candidate_cols = [c for c in candidate_cols if c in train.columns]
    categorical = [c for c in candidate_cols if not pd.api.types.is_numeric_dtype(train[c])]
    numeric = [c for c in candidate_cols if c not in categorical]
    analysis_cols = list(
        dict.fromkeys(
            numeric
            + [
                "nba_success_score",
                "is_star",
                "is_rotation_or_better",
                "is_not_nba_level",
            ]
        )
    )
    analysis = analyze_subset(train[analysis_cols].copy(), position)
    ranked = pd.DataFrame(columns=["feature", "selection_score"])
    if not analysis.empty:
        ranked = (
            analysis[
                analysis["analysis_type"].eq("raw")
                & analysis["target"].isin(
                    ["is_star", "is_rotation_or_better", "is_not_nba_level"]
                )
                & analysis["feature"].isin(numeric)
            ]
            .groupby("feature", as_index=False)
            .agg(
                selection_score=("rank_score", "mean"),
                mean_abs_spearman=("abs_spearman", "mean"),
                mean_mutual_information=("mutual_information", "mean"),
                targets_measured=("target", "nunique"),
            )
            .sort_values("selection_score", ascending=False)
        )

    selected_numeric = []
    for feature in ranked["feature"]:
        series = pd.to_numeric(train[feature], errors="coerce")
        redundant = False
        for kept in selected_numeric:
            corr = series.corr(pd.to_numeric(train[kept], errors="coerce"))
            if pd.notna(corr) and abs(corr) >= 0.90:
                redundant = True
                break
        if not redundant:
            selected_numeric.append(feature)
        if len(selected_numeric) >= max_numeric_features:
            break

    if len(selected_numeric) < min(8, len(numeric)):
        for feature in numeric:
            if feature not in selected_numeric:
                selected_numeric.append(feature)
            if len(selected_numeric) >= min(8, len(numeric)):
                break

    selected = selected_numeric + categorical
    ranked["position"] = position
    ranked["selected"] = ranked["feature"].isin(selected_numeric)
    ranked["selection_rank"] = np.arange(1, len(ranked) + 1)
    return selected, ranked


def rolling_draft_year_folds(train: pd.DataFrame, max_folds: int = 4):
    years = sorted(pd.to_numeric(train["draft_year"], errors="coerce").dropna().astype(int).unique())
    folds = []
    for year in years[-max_folds:]:
        fold_train = train[pd.to_numeric(train["draft_year"], errors="coerce") < year]
        fold_valid = train[pd.to_numeric(train["draft_year"], errors="coerce") == year]
        if (
            len(fold_train) >= 30
            and len(fold_valid) >= 5
            and fold_train[TARGET_COLUMN].nunique() >= 2
            and fold_valid[TARGET_COLUMN].nunique() >= 2
        ):
            folds.append((year, fold_train, fold_valid))
    return folds


def tune_position_algorithm(
    train: pd.DataFrame,
    cols: list[str],
    algorithm_name: str,
    base_estimator,
    position: str,
):
    grids = tuning_grids().get(algorithm_name, [{}])
    folds = rolling_draft_year_folds(train)
    tuning_rows = []
    for params in grids:
        fold_metrics = []
        for valid_year, fold_train, fold_valid in folds:
            try:
                X_train = fold_train[cols].copy()
                X_valid = fold_valid[cols].copy()
                estimator = clone(base_estimator).set_params(**params)
                pipe = Pipeline(
                    steps=[
                        ("preprocess", make_preprocessor(X_train)),
                        ("model", estimator),
                    ]
                )
                pipe.fit(X_train, fold_train[TARGET_COLUMN])
                labels = list(pipe.named_steps["model"].classes_)
                metrics, _ = evaluate_predictions(
                    fold_valid[TARGET_COLUMN], pipe.predict_proba(X_valid), labels
                )
                metrics["validation_year"] = valid_year
                fold_metrics.append(metrics)
            except Exception as exc:
                logger.warning(
                    "Tuning fold failed: position=%s algorithm=%s year=%s error=%s",
                    position,
                    algorithm_name,
                    valid_year,
                    exc,
                )
        if not fold_metrics:
            continue
        result = {
            "position": position,
            "algorithm": algorithm_name,
            "params": json.dumps(params, sort_keys=True),
            "cv_folds": len(fold_metrics),
            "cv_macro_f1": float(np.mean([m["macro_f1"] for m in fold_metrics])),
            "cv_balanced_accuracy": float(
                np.mean([m["balanced_accuracy"] for m in fold_metrics])
            ),
            "cv_accuracy": float(np.mean([m["accuracy"] for m in fold_metrics])),
        }
        result["cv_selection_score"] = (
            0.60 * result["cv_macro_f1"]
            + 0.30 * result["cv_balanced_accuracy"]
            + 0.10 * result["cv_accuracy"]
        )
        tuning_rows.append(result)

    if not tuning_rows:
        return clone(base_estimator), pd.DataFrame()
    tuning = pd.DataFrame(tuning_rows).sort_values("cv_selection_score", ascending=False)
    best_params = json.loads(tuning.iloc[0]["params"])
    return clone(base_estimator).set_params(**best_params), tuning


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

    leaked = sorted(set(cols) & set(OUTCOME_COLUMNS))
    if leaked:
        raise ValueError(
            f"NBA outcome leakage blocked for {model_name}: {', '.join(leaked)}"
        )

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


def select_best_position_models(metrics: pd.DataFrame) -> pd.DataFrame:
    """Select each position model, honoring any documented explicit override."""
    position_metrics = metrics[
        metrics["model_name"].astype(str).str.startswith("position_specific_")
    ].copy()
    if position_metrics.empty:
        return position_metrics

    position_metrics["position"] = position_metrics["model_name"].str.replace(
        "position_specific_", "", regex=False
    )
    position_metrics["cv_selection_score"] = pd.to_numeric(
        position_metrics.get("cv_selection_score"), errors="coerce"
    ).fillna(-np.inf)

    selected = []
    for position, group in position_metrics.groupby("position", sort=True):
        ranked = group.sort_values(
            ["cv_selection_score", "macro_f1"], ascending=[False, False]
        )
        override = POSITION_MODEL_OVERRIDES.get(str(position))
        override_rows = ranked[ranked["algorithm"].eq(override)] if override else ranked.iloc[0:0]
        chosen = (override_rows if not override_rows.empty else ranked).iloc[0].copy()
        chosen["selection_reason"] = (
            "configured_override" if not override_rows.empty else "rolling_validation"
        )
        selected.append(chosen)

    return pd.DataFrame(selected).rename(columns={"algorithm": "best_algorithm"})


def train_models(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = pd.read_csv(PROCESSED_FILES["training_labeled"])

    df = df[df[TARGET_COLUMN].notna()].copy()

    train, test = split_by_draft_year(df)

    rows = []
    tuning_outputs = []
    selected_feature_outputs = []
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

    # Position-specific models use train-only feature selection and rolling-year
    # hyperparameter tuning. The chronological test classes remain untouched.
    for pos in sorted(df.get("primary_position", pd.Series(dtype=str)).dropna().unique()):
        sub = df[df["primary_position"] == pos].copy()

        if len(sub) < MIN_POSITION_MODEL_SAMPLE:
            continue

        tr, te = split_by_draft_year(sub)

        if tr.empty or te.empty:
            continue

        base_cols = feature_sets(sub)["college_age_recruiting"]
        cols, feature_report = select_position_features(tr, base_cols, str(pos))
        selected_feature_outputs.append(feature_report)

        for algo_name, est in algos.items():
            try:
                tuned_estimator, tuning = tune_position_algorithm(
                    tr, cols, algo_name, est, str(pos)
                )
                if not tuning.empty:
                    tuning_outputs.append(tuning)
                result = train_one_model(
                    tr,
                    te,
                    cols,
                    algo_name,
                    tuned_estimator,
                    f"position_specific_{pos}",
                )
                if not tuning.empty:
                    result.update(
                        {
                            "cv_macro_f1": tuning.iloc[0]["cv_macro_f1"],
                            "cv_balanced_accuracy": tuning.iloc[0]["cv_balanced_accuracy"],
                            "cv_accuracy": tuning.iloc[0]["cv_accuracy"],
                            "cv_selection_score": tuning.iloc[0]["cv_selection_score"],
                            "best_params": tuning.iloc[0]["params"],
                        }
                    )
                rows.append(result)
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
    write_csv(
        pd.concat(tuning_outputs, ignore_index=True) if tuning_outputs else pd.DataFrame(),
        REPORTS_DIR / "position_model_tuning.csv",
    )
    write_csv(
        pd.concat(selected_feature_outputs, ignore_index=True)
        if selected_feature_outputs
        else pd.DataFrame(),
        REPORTS_DIR / "selected_features_by_position.csv",
    )

    best_positions = select_best_position_models(metrics)
    if not best_positions.empty:
        write_csv(best_positions, REPORTS_DIR / "best_model_by_position.csv")
        importance_outputs = []
        for _, best_row in best_positions.iterrows():
            destination = MODELS_DIR / f"best_model_{str(best_row['position']).lower()}.pkl"
            shutil.copyfile(best_row["model_path"], destination)
            artifact = joblib.load(destination)
            pos_subset = df[df["primary_position"].eq(best_row["position"])].copy()
            _, pos_test = split_by_draft_year(pos_subset)
            cols = artifact["feature_columns"]
            if pos_test.empty:
                continue
            try:
                importance = permutation_importance(
                    artifact["pipeline"],
                    pos_test[cols],
                    pos_test[TARGET_COLUMN],
                    scoring="f1_macro",
                    n_repeats=10,
                    random_state=RANDOM_STATE,
                    # Some native model libraries (notably LightGBM/OpenMP on
                    # macOS) are unsafe when joblib forks prediction workers.
                    n_jobs=1,
                )
                importance_outputs.append(
                    pd.DataFrame(
                        {
                            "position": best_row["position"],
                            "feature": cols,
                            "importance": importance.importances_mean,
                            "importance_std": importance.importances_std,
                            "algorithm": best_row["best_algorithm"],
                        }
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Permutation importance failed: position=%s error=%s",
                    best_row["position"],
                    exc,
                )
        write_csv(
            pd.concat(importance_outputs, ignore_index=True)
            if importance_outputs
            else pd.DataFrame(),
            REPORTS_DIR / "best_model_feature_importance_by_position.csv",
        )

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
