# TODO

## 1. Project Setup

- [x] Read project PDFs and understand assignment goal.
- [x] Inspect available Food.com dataset files.
- [x] Create scaffold folders for data work, modeling, reports, and outputs.
- [x] Create a 3-person work split.
- [ ] Confirm final deliverables required by the class: notebook, report, slides, code, or all of these.
- [ ] Decide whether ratings of `0` mean dislike, missing/implicit feedback, or should be filtered.

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
- [ ] Clean ratings according to the chosen `0`-rating policy.
- [ ] Remove or cap invalid cooking times.
- [ ] Parse list-like columns such as `tags`, `ingredients`, `nutrition`, and `steps`.
- [ ] Create a smaller development sample if full data is slow.
- [ ] Save cleaned/intermediate artifacts if useful.

## 4. Collaborative Filtering Baseline

- [ ] Train a simple global/user/item mean baseline.
- [ ] Train SVD collaborative filtering model using Surprise or an equivalent implementation.
- [ ] Tune SVD hyperparameters on validation data.
- [ ] Evaluate RMSE/MAE on validation and test sets.
- [ ] Generate top-N recommendations for sample users.
- [ ] Save baseline predictions and trained model artifact.

## 5. Feature Engineering

- [ ] Create recipe feature matrix from tags.
- [ ] Add nutrition features.
- [ ] Add cooking-time features.
- [ ] Add ingredient-based features or ingredient categories.
- [ ] Mark dietary tags such as vegetarian, vegan, gluten-free, low-carb, dessert, etc.
- [ ] Optionally infer broad cuisine categories from tags.

## 6. Bayesian Personalization

- [ ] Define the Bayesian preference representation.
- [ ] Choose initial priors for user preferences.
- [ ] Update posterior preferences from each user's ratings.
- [ ] Estimate uncertainty/confidence for recommendations.
- [ ] Test updating behavior on a few example users.
- [ ] Compare simple conjugate updating vs. PyMC model if time permits.

## 7. Hybrid Recommendation Model

- [ ] Combine SVD predicted rating with Bayesian preference score.
- [ ] Choose weighting strategy between collaborative and Bayesian components.
- [ ] Include confidence score in final recommendation output.
- [ ] Filter recipes using user constraints such as available ingredients or dietary restrictions if implemented.
- [ ] Produce final top-N recommendations with recipe names and explanations.

## 8. Evaluation

- [ ] Compare baseline SVD vs. hybrid Bayesian recommender.
- [ ] Evaluate RMSE/MAE for rating prediction.
- [ ] Evaluate ranking quality with Precision@K, Recall@K, or NDCG@K.
- [ ] Check whether confidence estimates are meaningful/calibrated.
- [ ] Include qualitative examples showing posterior updates.

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
- [ ] Add recipe explanations: "recommended because you liked Indian dishes and short cook times."
- [ ] Add ingredient availability filtering.
- [ ] Add dietary restriction filtering.
- [ ] Build a small Streamlit app demo.
