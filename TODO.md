# TODO

## 1. Project Setup

- [x] Read project PDFs and understand assignment goal.
- [x] Inspect available Food.com dataset files.
- [x] Create scaffold folders for data work, modeling, reports, and outputs.
- [x] Create a 3-person work split.
- [ ] Confirm final deliverables required by the class: notebook, report, slides, code, or all of these.
- [x] Decide whether ratings of `0` mean dislike, missing/implicit feedback, or should be filtered. (Policy: exclude from training/eval — `config.yaml` `zero_rating_policy`, asserted in hannah_02–04.)

## 2. Exploratory Data Analysis

- [ ] Summarize number of users, recipes, ratings, and sparsity.
- [ ] Plot rating distribution.
- [ ] Inspect recipe metadata columns: tags, ingredients, nutrition, minutes, steps.
- [ ] Identify missing values and bad records.
- [ ] Detect cooking-time outliers and decide cleaning rule.
- [ ] Examine most common tags/cuisines/dietary labels.
- [ ] Check user activity distribution and recipe popularity distribution.

## 3. Data Cleaning

- [ ] Load raw and preprocessed CSV files through reusable functions.
- [x] Clean ratings according to the chosen `0`-rating policy. (rating==0 excluded in hannah_02–04.)
- [ ] Remove or cap invalid cooking times.
- [x] Parse list-like columns such as `tags`, `ingredients`, `nutrition`, and `steps`. (`src/features/recipe_features.py` + `src/features/hannah_features.py`; used in hannah_03/04.)
- [ ] Create a smaller development sample if full data is slow.
- [ ] Save cleaned/intermediate artifacts if useful.

## 4. Collaborative Filtering Baseline

- [x] Train a simple global/user/item mean baseline. (hannah_02; incl. Surprise BaselineOnly ablation.)
- [x] Train SVD collaborative filtering model using Surprise or an equivalent implementation. (hannah_02.)
- [x] Tune SVD hyperparameters on validation data. (hannah_02 inner-holdout grid search; winner in `outputs/hannah_svd_params.json`.)
- [x] Evaluate RMSE/MAE on validation and test sets. (hannah_02: warm random/temporal holdouts + cold validation/test.)
- [x] Generate top-N recommendations for sample users. (hannah_02 ranking eval; hannah_04 top-N demos.)
- [x] Save baseline predictions and trained model artifact. (`outputs/hannah_svd_preds_*.csv`, `outputs/models/hannah_svd.pkl`, `outputs/models/hannah_baseline_only.pkl`.)

## 5. Feature Engineering

- [x] Create recipe feature matrix from tags. (`src/features/hannah_features.py` `build_recipe_flags`; used in hannah_03/04.)
- [x] Add nutrition features. (Calorie terciles cal_low/mid/high in `hannah_features`.)
- [x] Add cooking-time features. (Time tag family: 15/30/60-minutes-or-less, 4-hours-or-less.)
- [x] Add ingredient-based features or ingredient categories. (8 ingredient-keyword flags in `hannah_features`.)
- [x] Mark dietary tags such as vegetarian, vegan, gluten-free, low-carb, dessert, etc. (Dietary + dish families in `hannah_features`.)
- [x] Optionally infer broad cuisine categories from tags. (Cuisine family in `hannah_features`.)

## 6. Bayesian Personalization

- [x] Define the Bayesian preference representation. (Beta–Binomial per user/family/tag; `src/models/bayesian_updater.py`, hannah_03.)
- [x] Choose initial priors for user preferences. (Empirical-Bayes priors + strength sensitivity in hannah_03.)
- [x] Update posterior preferences from each user's ratings. (Sequential replay + per-tag partially pooled posteriors, hannah_03.)
- [x] Estimate uncertainty/confidence for recommendations. (CI-width confidence in `bayesian_updater`; used in hannah_03/04.)
- [x] Test updating behavior on a few example users. (hannah_03 example-user posterior evolution.)
- [x] Compare simple conjugate updating vs. PyMC model if time permits. (hannah_03 PyMC hierarchical comparison.)

## 7. Hybrid Recommendation Model

- [x] Combine SVD predicted rating with Bayesian preference score. (hannah_04 hybrid blends, stars + probability tracks.)
- [x] Choose weighting strategy between collaborative and Bayesian components. (hannah_04 weight sweep chosen on cold validation.)
- [x] Include confidence score in final recommendation output. (hannah_04 top-N output incl. calibrated confidence.)
- [x] Filter recipes using user constraints such as available ingredients or dietary restrictions if implemented. (hannah_04 constrained demo → `outputs/recommendations/hannah_topn_constrained.csv`.)
- [x] Produce final top-N recommendations with recipe names and explanations. (hannah_04 top-N with per-tag explanations.)

## 8. Evaluation

- [x] Compare baseline SVD vs. hybrid Bayesian recommender. (hannah_04; bias-only ablation in hannah_02/04.)
- [x] Evaluate RMSE/MAE for rating prediction. (hannah_02/04 across warm random, warm temporal, cold val/test.)
- [x] Evaluate ranking quality with Precision@K, Recall@K, or NDCG@K. (HR@10/NDCG@10 with CIs, 1+99 protocol — hannah_02/04.)
- [x] Check whether confidence estimates are meaningful/calibrated. (hannah_04 calibration table → `outputs/hannah_calibration.csv`.)
- [x] Include qualitative examples showing posterior updates. (hannah_03 example users; hannah_04 explanations.)

## 9. Final Report / Presentation

- [ ] Write problem motivation.
- [ ] Describe dataset and preprocessing.
- [ ] Explain SVD collaborative filtering.
- [ ] Explain Bayesian updating and uncertainty.
- [ ] Present model results and plots.
- [ ] Include example recommendations.
- [ ] Discuss limitations and future improvements.

## 10. Stretch Goals

- [ ] Build a simple CLI/demo script for entering a user ID and returning recommendations.
- [x] Add recipe explanations: "recommended because you liked Indian dishes and short cook times." (hannah_04 per-tag posterior explanations.)
- [x] Add ingredient availability filtering. (hannah_04 pantry-constrained demo.)
- [x] Add dietary restriction filtering. (hannah_04 vegetarian-constrained demo.)
- [ ] Build a small Streamlit app demo.
