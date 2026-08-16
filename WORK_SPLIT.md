# 3-Person Work Split

This project can be divided into three mostly independent tracks. Each person should own their track, produce clear outputs, and then meet at the hybrid integration step.

## Team Roles

| Person | Main Role | Main Deliverable |
| --- | --- | --- |
| Person 1 | Data + EDA | Clean dataset, exploratory plots, feature-ready recipe/user data |
| Person 2 | Collaborative Filtering | SVD baseline model, validation/test metrics, top-N baseline recommendations |
| Person 3 | Bayesian + Hybrid Model | Bayesian preference updater, uncertainty scores, final hybrid recommender |

Replace `Person 1`, `Person 2`, and `Person 3` with actual names once assigned.

## Person 1: Data Cleaning, EDA, and Features

### Goal

Understand and prepare the Food.com dataset so the models have clean inputs.

### Main Files

- `notebooks/final/01_data_exploration.ipynb`
- `src/data/load_data.py`
- `src/data/cleaning.py`
- `src/features/recipe_features.py`
- `reports/figures/`

### Tasks

- [ ] Summarize dataset size: users, recipes, ratings, sparsity.
- [ ] Plot rating distribution.
- [ ] Investigate whether rating `0` should be removed or treated as negative feedback.
- [ ] Inspect missing values in recipes/interactions.
- [ ] Clean extreme cooking-time outliers.
- [ ] Parse recipe tags, ingredients, and nutrition.
- [ ] Create basic recipe features:
  - vegetarian/dietary indicators
  - dessert/main dish/etc. indicators
  - broad cuisine tags if possible
  - cooking-time buckets
  - nutrition features
- [ ] Save useful plots to `reports/figures/`.
- [ ] Document cleaning decisions in `docs/project_brief.md` or notebook markdown.

### Output for Team

- Cleaned/explained data assumptions.
- Feature columns that Person 3 can use for Bayesian updating.
- EDA figures for final report/slides.

## Person 2: Collaborative Filtering Baseline

### Goal

Build the recommendation baseline using historical user-recipe ratings.

### Main Files

- `notebooks/final/02_svd_baseline.ipynb`
- `src/models/baseline.py`
- `src/models/svd_model.py`
- `src/evaluation/metrics.py`
- `outputs/models/`
- `outputs/recommendations/`

### Tasks

- [ ] Train a global mean/user mean/item mean baseline.
- [ ] Train an SVD model using `interactions_train.csv`.
- [ ] Evaluate on validation and test sets.
- [ ] Report RMSE and MAE.
- [ ] Tune simple SVD hyperparameters if time allows.
- [ ] Generate top-N recommendations for a few sample users.
- [ ] Save model or prediction outputs in `outputs/`.
- [ ] Write a short explanation of how SVD answers: "What recipes would similar users like?"

### Output for Team

- Baseline model performance.
- SVD prediction function or saved predictions.
- Baseline recommendation examples for final comparison.

## Person 3: Bayesian Updating and Hybrid Recommender

### Goal

Add Bayesian personalization and uncertainty to the recommender.

### Main Files

- `notebooks/final/03_bayesian_updating.ipynb`
- `notebooks/final/04_hybrid_recommender.ipynb`
- `src/models/bayesian_updater.py`
- `src/recommender/hybrid.py`
- `docs/modeling_plan.md`

### Tasks

- [ ] Define what counts as positive feedback, likely rating `>= 4`.
- [ ] Create Bayesian preference priors for recipe features/tags.
- [ ] Update user preferences after ratings.
- [ ] Compute posterior mean preference scores.
- [ ] Compute uncertainty/confidence scores.
- [ ] Combine Bayesian score with SVD score.
- [ ] Generate final top-N recommendations.
- [ ] Add simple explanations, for example:
  - "Recommended because this matches tags you tend to rate highly."
  - "High confidence because the user has rated many similar recipes."
- [ ] Compare hybrid recommendations to SVD-only recommendations.

### Output for Team

- Bayesian updating demo.
- Hybrid scoring function.
- Final recommendation examples with confidence.

## Shared Integration Plan

### Integration Meeting 1: After EDA and Baseline

Everyone should agree on:

- [ ] Whether to remove rating `0`.
- [ ] Whether to use raw IDs or internal `u`/`i` IDs.
- [ ] Which recipe features are reliable enough for Bayesian updating.
- [ ] Which metric will be the headline result.

### Integration Meeting 2: Before Final Report

Everyone should bring:

- [ ] 1-2 plots or tables from their part.
- [ ] 1 paragraph explaining their method.
- [ ] 1 paragraph explaining results/limitations.
- [ ] Any code that the final notebook depends on.

## Suggested Final Deliverable Ownership

| Section | Owner |
| --- | --- |
| Introduction / motivation | Shared |
| Dataset description | Person 1 |
| EDA and preprocessing | Person 1 |
| Collaborative filtering method | Person 2 |
| SVD results | Person 2 |
| Bayesian method | Person 3 |
| Hybrid recommendation results | Person 3 |
| Limitations / future work | Shared |

## File Ownership Rules

To avoid merge confusion:

- Person 1 owns `01_data_exploration.ipynb`, `src/data/`, and `src/features/`.
- Person 2 owns `notebooks/final/02_svd_baseline.ipynb`, `src/models/baseline.py`, and `src/models/svd_model.py`.
- Person 3 owns `notebooks/final/03_bayesian_updating.ipynb`, `notebooks/final/04_hybrid_recommender.ipynb`, `src/models/bayesian_updater.py`, and `src/recommender/`.
- Anyone can edit docs, but mention the change in the group chat.
- Avoid editing another person's notebook unless they ask.

## Minimum Viable Project

If time gets tight, finish these first:

- [ ] Clean ratings and remove obvious bad recipe records.
- [ ] Train SVD baseline.
- [ ] Build simple Beta-Binomial Bayesian preference updater over recipe tags.
- [ ] Combine scores with a weighted average.
- [ ] Show example recommendations with confidence.
- [ ] Report RMSE/MAE and at least one qualitative recommendation example.
