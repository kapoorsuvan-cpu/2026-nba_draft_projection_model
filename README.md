# NBA College Projection Model

This project predicts NBA player outcomes from final-season NCAA production, draft position, age, recruiting, position, and team context.

The repository keeps two dashboard apps:

- `frontend/` preserves the prior first-four-season experiment.
- `frontend-career-outcomes/` presents the post-rookie career-outcome experiment with tuned G/F/C models.

The project is now **API-first**. Raw CSV files are used as local caches only. If the files in `data/raw/` do not exist, the pipeline fetches the data from APIs and writes those cache files automatically.

## Data sources

### 1. CollegeBasketballData / `cbbd`

Used for:

- NCAA player season stats
- NCAA player shooting stats
- NCAA team season stats / team context
- Recruiting rankings

You need a free CollegeBasketballData key.

Get one here:

```text
https://collegebasketballdata.com/key
```

Add it to `.env`:

```bash
CBBD_API_KEY=
```

The code also accepts:

```bash
BEARER_TOKEN=
```

Relevant files:

```text
src/config.py
src/data_sources.py
```

### 2. `nba_api`

Used for:

- NBA draft history
- NBA player career stats
- career NBA outcome labels with post-rookie role validation
- All-Star / All-NBA award indicators when available

`nba_api` does **not** require an API key.

## What CSVs do I need now?

For training: **none manually**, assuming your API key works.

The pipeline will create these cache files automatically:

```text
data/raw/historical_college_stats.csv
data/raw/historical_draft_results.csv
data/raw/historical_nba_outcomes.csv
data/raw/recruiting_rankings.csv
```

The ESPN Top 100 CSV is **not needed for training**. It is only for future current-prospect prediction:

```text
data/raw/espn_2026_top100_raw.csv
```

## Training years

The model uses only draft classes with an observable fifth NBA season.

```text
Eligible draft classes: 2006-2021
Train split: 2006-2019
Test split: 2020-2021
Excluded from training labels: 2022+
```

This avoids using recent players whose NBA outcome is still incomplete.

## Outcome definitions

- **Star:** made an All-Star team, made an All-NBA team, or received a max contract at any point in his career.
- **Rotation:** completed at least two seasons with 40+ games and 15+ MPG, with at least one of those seasons occurring in NBA year five or later, and did not meet the Star definition.
- **Not NBA Level:** met neither benchmark.

NBA outcomes are used only to create the target label; they are never model inputs. The API-built
dataset includes All-Star and All-NBA awards. Because `nba_api` does not publish contract values,
`max_contract_indicator` defaults to zero unless a contract-enriched NBA outcome cache supplies it.

## Install

```bash
cd nba_college_projection_model
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Then add your CollegeBasketballData key:

```bash
CBBD_API_KEY=your_key_here
```

## Run the pipeline

```bash
python run_pipeline.py
```

The pipeline will:

1. Fetch/cache draft history from `nba_api`
2. Fetch/cache college player stats from CollegeBasketballData
3. Fetch/cache NBA career outcomes from `nba_api`
4. Fetch/cache recruiting rankings from CollegeBasketballData
5. Build one row per drafted NCAA player using final college season
6. Create labels
7. Run position-specific correlation analysis
8. Train and compare models
9. Select/evaluate the best model

Google TabFM 1.0.0 is evaluated on the same chronological holdout and canonical
full feature set. It is adopted only when its holdout accuracy is strictly higher than
every non-TabFM candidate.

### Position-model methodology

1. Split draft classes chronologically before any feature selection.
2. On training classes only, rank safe college/recruiting features separately for G, F, and C using Pearson correlation, Spearman correlation, and mutual information against the outcome targets.
3. Remove near-duplicate numeric features with absolute correlation of 0.90 or greater and retain the strongest non-redundant position-specific features.
4. Tune logistic regression, random forest, histogram gradient boosting, LightGBM, and XGBoost with rolling draft-year validation inside the training period.
5. Select each position algorithm using 60% macro F1, 30% balanced accuracy, and 10% accuracy across the rolling validation folds.
6. Score the chosen configurations once on the untouched 2020-2021 test classes.

The correlation reports are therefore operational inputs to position feature selection, not just dashboard diagnostics. NBA outcome fields remain excluded from every model feature set.

Because the center holdout contains only ten players and produced sharply conflicting model results, `POSITION_MODEL_OVERRIDES` currently selects Random Forest for centers. Guards and forwards remain selected strictly by rolling validation.

## Important limitations

- `nba_api` does not provide Basketball Reference-style VORP/BPM/Win Shares or contract values. Contract-enriched input can supply `max_contract_indicator`; otherwise it defaults to zero.
- Second contract is inferred as appearing in a fifth NBA season after draft. That is a practical proxy, not a contract feed.
- Recruiting is only as complete as CollegeBasketballData's recruiting endpoint.
- Position, height, and class year may be incomplete if the source does not provide them for a player-season.
- ESPN Top 100 is intentionally optional and is not part of model training.

## Key config values

Edit these in `src/config.py`:

```python
USE_API_DATA_SOURCES = True
REFRESH_API_CACHE = False
ELIGIBLE_DRAFT_MIN_YEAR = 2006
ELIGIBLE_DRAFT_MAX_YEAR = 2021
TRAIN_MAX_DRAFT_YEAR = 2019
TEST_MIN_DRAFT_YEAR = 2020
TEST_MAX_DRAFT_YEAR = 2021
RUN_2026_PROSPECT_PIPELINE_BY_DEFAULT = False
```

To force a re-fetch from APIs:

```python
REFRESH_API_CACHE = True
```

## Output files

Processed training data:

```text
data/processed/model_training_dataset.csv
data/processed/model_training_dataset_with_labels.csv
```

Model outputs:

```text
models/best_model.pkl
models/best_model_metadata.json
```

Reports:

```text
reports/model_comparison_overall.csv
reports/model_comparison_by_position.csv
reports/model_comparison_by_position_group.csv
reports/correlation_by_position_pg.csv
reports/correlation_by_position_sg.csv
reports/correlation_by_position_sf.csv
reports/correlation_by_position_pf.csv
reports/correlation_by_position_c.csv
```
