from src.utils import setup_logging
from src.build_training_dataset import build_training_dataset
from src.analyze_correlations_by_position import run_position_correlation_analysis
from src.train_model import train_models
from src.compare_models import compare_and_select_best
from src.evaluate_model import evaluate_best_model
from src.build_2026_prospect_dataset import build_2026_prospect_dataset
from src.predict_2026_class import predict_2026_class
from src.config import RAW_FILES, RUN_2026_PROSPECT_PIPELINE_BY_DEFAULT, TRAIN_MAX_DRAFT_YEAR


def main():
    logger = setup_logging()
    logger.info("1/7 Building historical training dataset")
    training = build_training_dataset()
    logger.info("2/7 Running train-only position-level correlation analysis")
    analysis_train = training[
        training["draft_year"].astype(float) <= TRAIN_MAX_DRAFT_YEAR
    ].copy()
    run_position_correlation_analysis(analysis_train)
    logger.info("3/7 Training model suite")
    metrics = train_models(training)
    logger.info("4/7 Selecting best model")
    compare_and_select_best(metrics)
    logger.info("5/7 Evaluating best model")
    evaluate_best_model(training)
    if RUN_2026_PROSPECT_PIPELINE_BY_DEFAULT and RAW_FILES["espn_2026"].exists():
        logger.info("6/7 Building 2026 prospect dataset")
        prospects = build_2026_prospect_dataset()
        logger.info("7/7 Predicting 2026 class")
        predict_2026_class(prospects)
    else:
        logger.info("Skipping 2026 prospect prediction. This is optional and not needed for model training.")
        logger.info("To run it later, add data/raw/espn_2026_top100_raw.csv and set RUN_2026_PROSPECT_PIPELINE_BY_DEFAULT=True in src/config.py")
    logger.info("Pipeline complete")


if __name__ == "__main__":
    main()
