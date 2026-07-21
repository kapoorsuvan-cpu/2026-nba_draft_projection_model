import logging
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.linear_model import LinearRegression
from .config import PROCESSED_FILES, REPORTS_DIR, SPECIFIC_POSITIONS, POSITION_GROUPS, MIN_POSITION_SAMPLE
from .features import get_feature_columns
from .utils import write_csv

logger = logging.getLogger(__name__)

TARGETS = ["nba_success_score", "is_star", "is_rotation_or_better", "is_not_nba_level"]
CONTROLS = {
    "raw": [],
    "without_draft_pick": [],
    "control_draft_pick": ["draft_pick_overall"],
    "control_age": ["age_final_college_season"],
    "control_recruiting_rank": ["recruiting_rank"],
    "control_draft_age_recruiting": ["draft_pick_overall", "age_final_college_season", "recruiting_rank"],
}


def _residualize(y: pd.Series, controls: pd.DataFrame) -> pd.Series:
    tmp = pd.concat([y, controls], axis=1).dropna()
    if tmp.shape[0] < 10 or controls.shape[1] == 0:
        return y
    model = LinearRegression().fit(tmp.iloc[:, 1:], tmp.iloc[:, 0])
    resid = tmp.iloc[:, 0] - model.predict(tmp.iloc[:, 1:])
    out = pd.Series(index=y.index, dtype=float)
    out.loc[tmp.index] = resid
    return out


def _mi(x: pd.Series, y: pd.Series, discrete_target: bool) -> float:
    tmp = pd.concat([x, y], axis=1).dropna()
    if tmp.shape[0] < 20 or tmp.iloc[:, 0].nunique() < 2 or tmp.iloc[:, 1].nunique() < 2:
        return np.nan
    X = tmp.iloc[:, [0]]
    yy = tmp.iloc[:, 1]
    try:
        return float(mutual_info_classif(X, yy.astype(int), random_state=42)[0]) if discrete_target else float(mutual_info_regression(X, yy, random_state=42)[0])
    except Exception:
        return np.nan


def analyze_subset(df: pd.DataFrame, subset_name: str) -> pd.DataFrame:
    # pandas cannot concatenate an empty Series with a zero-column DataFrame
    # whose index has a different inferred dtype. Empty legacy position groups
    # should simply produce an empty report.
    if df.empty:
        return pd.DataFrame(columns=[
            "subset", "sample_size", "small_sample_warning", "target",
            "analysis_type", "feature", "pearson", "spearman",
            "mutual_information", "n_pairwise", "abs_spearman", "rank_score",
        ])
    # Correlate outcomes only against information known before the NBA career.
    # get_feature_columns applies the same authoritative leakage boundary used
    # by model training.
    safe_features = get_feature_columns(
        df, include_draft_pick=True, include_position=False
    )
    features = [
        c for c in safe_features
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    rows = []
    n = len(df)
    warning = n < MIN_POSITION_SAMPLE
    for target in TARGETS:
        if target not in df.columns:
            continue
        discrete = target != "nba_success_score"
        for analysis, controls in CONTROLS.items():
            use_features = features.copy()
            if analysis == "without_draft_pick":
                use_features = [f for f in use_features if f not in {"draft_pick_overall", "draft_round"}]
            target_series = pd.to_numeric(df[target], errors="coerce")
            control_df = df[[c for c in controls if c in df.columns]].apply(pd.to_numeric, errors="coerce") if controls else pd.DataFrame(index=df.index)
            y = _residualize(target_series, control_df) if controls else target_series
            for feat in use_features:
                if feat in TARGETS or feat in controls:
                    continue
                x = pd.to_numeric(df[feat], errors="coerce")
                if controls:
                    x = _residualize(x, control_df)
                tmp = pd.concat([x, y], axis=1).dropna()
                if tmp.shape[0] < 10 or tmp.iloc[:,0].nunique() < 2:
                    continue
                rows.append({
                    "subset": subset_name,
                    "sample_size": n,
                    "small_sample_warning": warning,
                    "target": target,
                    "analysis_type": analysis,
                    "feature": feat,
                    "pearson": tmp.iloc[:,0].corr(tmp.iloc[:,1], method="pearson"),
                    "spearman": tmp.iloc[:,0].corr(tmp.iloc[:,1], method="spearman"),
                    "mutual_information": _mi(tmp.iloc[:,0], tmp.iloc[:,1], discrete),
                    "n_pairwise": len(tmp),
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["abs_spearman"] = out["spearman"].abs()
        out["rank_score"] = out[["pearson", "spearman", "mutual_information"]].abs().mean(axis=1)
    return out


def run_position_correlation_analysis(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = pd.read_csv(PROCESSED_FILES["training_labeled"])
    outputs = []
    for pos in SPECIFIC_POSITIONS:
        sub = df[df.get("primary_position", "").astype(str).str.upper() == pos].copy()
        res = analyze_subset(sub, pos)
        write_csv(res, REPORTS_DIR / f"correlation_by_position_{pos.lower()}.csv")
        if not res.empty:
            outputs.append(res)
    for group in POSITION_GROUPS:
        sub = df[df.get("position_group", "").astype(str).str.lower() == group].copy()
        res = analyze_subset(sub, group)
        write_csv(res, REPORTS_DIR / f"correlation_by_position_group_{group}.csv")
        if not res.empty:
            outputs.append(res)
    summary = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
    if not summary.empty:
        summary = summary.sort_values(["subset", "target", "analysis_type", "rank_score"], ascending=[True, True, True, False])
    write_csv(summary, REPORTS_DIR / "correlation_by_position_summary.csv")
    return summary

if __name__ == "__main__":
    run_position_correlation_analysis()
