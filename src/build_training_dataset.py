import logging
import pandas as pd
from .config import INTERIM_FILES, PROCESSED_FILES, ELIGIBLE_DRAFT_MIN_YEAR, ELIGIBLE_DRAFT_MAX_YEAR
from .clean_college_data import clean_college_data, select_final_college_seasons
from .clean_nba_data import clean_nba_data
from .clean_recruiting_data import clean_recruiting_data
from .data_sources import load_draft_results
from .features import create_engineered_features
from .label_success import create_success_labels
from .utils import normalize_name, fuzzy_merge, write_csv, create_unmatched_report

logger = logging.getLogger(__name__)


def clean_draft_data(draft: pd.DataFrame | None = None) -> pd.DataFrame:
    if draft is None:
        draft = load_draft_results()
    draft = draft.copy()
    draft.columns = [c.strip().lower().replace(" ", "_") for c in draft.columns]
    if "player_name" not in draft.columns:
        for c in ["player", "name"]:
            if c in draft.columns:
                draft["player_name"] = draft[c]
                break
    draft["normalized_player_name"] = draft["player_name"].map(normalize_name)
    for c in ["draft_year", "draft_round", "draft_pick_overall"]:
        if c in draft.columns:
            draft[c] = pd.to_numeric(draft[c], errors="coerce")
    return draft


def filter_eligible_draft_years(draft: pd.DataFrame) -> pd.DataFrame:
    """Keep only classes with enough NBA history for first-four-year labels."""
    draft = draft.copy()
    if "draft_year" not in draft.columns:
        return draft
    draft["draft_year"] = pd.to_numeric(draft["draft_year"], errors="coerce")
    before = len(draft)
    draft = draft[
        draft["draft_year"].between(ELIGIBLE_DRAFT_MIN_YEAR, ELIGIBLE_DRAFT_MAX_YEAR, inclusive="both")
    ].copy()
    logger.info(
        "Filtered draft data from %s rows to %s rows using eligible draft years %s-%s",
        before, len(draft), ELIGIBLE_DRAFT_MIN_YEAR, ELIGIBLE_DRAFT_MAX_YEAR
    )
    return draft


def build_training_dataset() -> pd.DataFrame:
    draft = filter_eligible_draft_years(clean_draft_data())
    college = clean_college_data()
    final_college = select_final_college_seasons(college, draft)
    nba = clean_nba_data()
    recruiting = clean_recruiting_data()

    # Anchor on drafted players; merge final college season, recruiting, and NBA outcomes.
    base = draft.copy()
    merged_college, rep_college = fuzzy_merge(base, final_college, left_year_col="draft_year", right_year_col="ncaa_season", year_tolerance=1, score_cutoff=88, suffix="_college")

    if not recruiting.empty:
        merged_recruiting, rep_recruiting = fuzzy_merge(merged_college, recruiting, score_cutoff=88, suffix="_recruiting")
    else:
        merged_recruiting, rep_recruiting = merged_college, pd.DataFrame()

    merged_nba, rep_nba = fuzzy_merge(merged_recruiting, nba, left_year_col="draft_year", right_year_col="draft_year" if "draft_year" in nba.columns else None, year_tolerance=0, score_cutoff=88, suffix="_nba")

    # Resolve duplicated columns from fuzzy joins.
    df = merged_nba.copy()
    for col in list(df.columns):
        if col.endswith("_college") and col[:-8] not in df.columns:
            df[col[:-8]] = df[col]
        if col.endswith("_recruiting") and col[:-12] not in df.columns:
            df[col[:-12]] = df[col]
        if col.endswith("_nba") and col[:-4] not in df.columns:
            df[col[:-4]] = df[col]

    df = create_engineered_features(df, save_dictionary=True)
    write_csv(df, PROCESSED_FILES["training"])
    labeled = create_success_labels(df)
    write_csv(labeled, PROCESSED_FILES["training_labeled"])
    create_unmatched_report({"draft_to_college": rep_college, "draft_to_recruiting": rep_recruiting, "draft_to_nba": rep_nba}, PROCESSED_FILES["unmatched"])
    logger.info("Saved training dataset with labels: %s rows", len(labeled))
    return labeled

if __name__ == "__main__":
    build_training_dataset()
