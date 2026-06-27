import logging
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process


def setup_logging(name: str = "nba_college_projection") -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)


def normalize_name(name) -> str:
    if pd.isna(name):
        return ""
    text = str(name).strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\b(jr|sr|ii|iii|iv|v)\.?\b", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_team(team) -> str:
    if pd.isna(team):
        return ""
    text = str(team).strip().lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def safe_read_csv(path: Path, required: bool = True, **kwargs) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        msg = f"Missing file: {path}. Add it to data/raw or update config.py."
        if required:
            raise FileNotFoundError(msg)
        logging.warning(msg)
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def coalesce_columns(df: pd.DataFrame, candidates: Iterable[str], target: str, default=np.nan) -> pd.DataFrame:
    existing = [c for c in candidates if c in df.columns]
    if not existing:
        df[target] = default
        return df
    out = df[existing[0]].copy()
    for col in existing[1:]:
        out = out.combine_first(df[col])
    df[target] = out
    return df


def parse_height_to_inches(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip().lower().replace('"', '').replace("’", "'")
    m = re.match(r"^(\d+)\s*[-']\s*(\d+)$", text)
    if m:
        return int(m.group(1)) * 12 + int(m.group(2))
    m = re.match(r"^(\d+)\s*ft\s*(\d+)?", text)
    if m:
        return int(m.group(1)) * 12 + int(m.group(2) or 0)
    return pd.to_numeric(value, errors="coerce")


def standardize_position(pos) -> tuple[str, str, str]:
    """
    Standardize raw position without pretending broad G/F labels are specific.

    Important:
    - G stays G
    - F stays F
    - PG/SG/SF/PF/C stay specific when available
    """
    if pd.isna(pos) or str(pos).strip() == "":
        return "", "", "unknown"

    raw = str(pos).upper().replace("-", "/").replace(",", "/").replace(" ", "")
    parts = [p for p in raw.split("/") if p]

    aliases = {
        "POINTGUARD": "PG",
        "SHOOTINGGUARD": "SG",
        "SMALLFORWARD": "SF",
        "POWERFORWARD": "PF",
        "CENTER": "C",
        "CENTRE": "C",
        "FC": "F/C",
        "GF": "G/F",
        "CG": "C/G",
    }

    cleaned = []
    for p in parts:
        mapped = aliases.get(p, p)
        if "/" in mapped:
            cleaned.extend([x for x in mapped.split("/") if x])
        else:
            cleaned.append(mapped)

    valid_order = ["PG", "SG", "SF", "PF", "C", "G", "F"]
    valid = [p for p in cleaned if p in valid_order]

    primary = valid[0] if valid else cleaned[0]
    secondary = valid[1] if len(valid) > 1 else ""
    group = position_group_from_primary(primary)

    return primary, secondary, group


def position_group_from_primary(primary: str, height=None) -> str:
    p = str(primary).upper()
    if p in {"PG", "SG"}:
        return "guard"
    if p == "SF":
        return "wing"
    if p == "PF":
        # Some PFs are wings; use height when available as a rough rule.
        h = parse_height_to_inches(height) if height is not None else np.nan
        return "wing" if pd.notna(h) and h <= 80 else "big"
    if p == "C":
        return "big"
    return "unknown"


def fuzzy_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_name_col: str = "normalized_player_name",
    right_name_col: str = "normalized_player_name",
    left_year_col: Optional[str] = None,
    right_year_col: Optional[str] = None,
    year_tolerance: int = 0,
    score_cutoff: int = 90,
    suffix: str = "_right",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fuzzy merge by normalized name, optionally constrained by draft/season year."""
    if right.empty or left.empty:
        merged = left.copy()
        report = left[[left_name_col]].copy() if left_name_col in left else pd.DataFrame()
        report["match_status"] = "no_right_data"
        return merged, report

    right = right.copy().reset_index(drop=True)
    right_choices = right[right_name_col].fillna("").tolist()
    matches = []
    for idx, row in left.reset_index(drop=True).iterrows():
        name = row.get(left_name_col, "")
        candidates = right
        choices = right_choices
        if left_year_col and right_year_col and pd.notna(row.get(left_year_col)):
            y = int(row[left_year_col])
            mask = (right[right_year_col].fillna(-9999).astype(int) - y).abs() <= year_tolerance
            candidates = right.loc[mask]
            choices = candidates[right_name_col].fillna("").tolist()
        if not choices:
            matches.append((idx, None, 0))
            continue
        best = process.extractOne(name, choices, scorer=fuzz.token_sort_ratio, score_cutoff=score_cutoff)
        if best:
            match_name, score, local_idx = best
            right_idx = candidates.index[local_idx]
            matches.append((idx, right_idx, score))
        else:
            matches.append((idx, None, 0))

    left_tmp = left.reset_index(drop=True).copy()
    left_tmp["_left_row_id"] = range(len(left_tmp))
    match_df = pd.DataFrame(matches, columns=["_left_row_id", "_right_row_id", "match_score"])
    right_tmp = right.copy()
    right_tmp["_right_row_id"] = right_tmp.index
    merged = left_tmp.merge(match_df, on="_left_row_id", how="left").merge(
        right_tmp, on="_right_row_id", how="left", suffixes=("", suffix)
    )
    
    score_col = f"match_score{suffix}" if suffix else "match_score"

    # If pandas created duplicate match-score columns from prior fuzzy merges,
    # preserve the current merge's score under a suffix-specific name.
    if "match_score" in merged.columns:
        merged = merged.rename(columns={"match_score": score_col})
    elif "match_score_y" in merged.columns:
        merged = merged.rename(columns={"match_score_y": score_col})
    elif score_col not in merged.columns:
        merged[score_col] = None

    # Clean up duplicate score columns left over from previous merges.
    for extra_col in ["match_score_x", "match_score_y"]:
        if extra_col in merged.columns and extra_col != score_col:
            merged = merged.drop(columns=[extra_col])

    report_cols = [c for c in ["_left_row_id", left_name_col, score_col] if c in merged.columns]
    report = merged[report_cols].copy()

    report["match_status"] = np.where(merged["_right_row_id"].notna(), "matched", "unmatched")
    return merged.drop(columns=["_left_row_id", "_right_row_id"], errors="coerce"), report


def create_unmatched_report(reports: dict[str, pd.DataFrame], output_path: Path) -> pd.DataFrame:
    frames = []
    for merge_name, df in reports.items():
        if df is None or df.empty:
            continue
        tmp = df.copy()
        tmp["merge_name"] = merge_name
        frames.append(tmp)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    write_csv(out, output_path)
    return out


def numeric_columns(df: pd.DataFrame, exclude: Iterable[str] = ()) -> list[str]:
    exclude = set(exclude)
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
