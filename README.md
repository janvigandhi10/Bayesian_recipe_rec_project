# Bayesian Personalized Meal Recommendation System

This project builds a hybrid recipe recommender for the Food.com Recipes and User Interactions dataset. The goal is to recommend recipes a user is likely to enjoy while also reporting uncertainty or confidence in each recommendation.

## Project Idea

The system combines two pieces:

1. **Collaborative filtering baseline**
   - Train an SVD recommender on historical user-recipe ratings.
   - Use it to estimate how much a user may like an unseen recipe.

2. **Bayesian personalization layer**
   - Represent user preferences as updateable probability distributions.
   - Update those preferences when new ratings or feedback arrive.
   - Adjust recommendation scores using recipe features such as tags, ingredients, nutrition, and cooking time.
   - Return both a recommendation score and a confidence/uncertainty estimate.

## Data

The project uses the Food.com dataset already stored in `Data/`.

Key files:

- `Data/RAW_recipes.csv`: raw recipe metadata, ingredients, tags, nutrition, steps, and cooking time
- `Data/RAW_interactions.csv`: raw user ratings and reviews
- `Data/interactions_train.csv`: training user-recipe ratings
- `Data/interactions_validation.csv`: validation ratings
- `Data/interactions_test.csv`: test ratings
- `Data/PP_recipes.csv`: preprocessed recipe representations
- `Data/PP_users.csv`: preprocessed user representations
- `Data/ingr_map.pkl`: ingredient mapping

## Repository Structure

```text
Project_bayesian/
├── Data/                       # Provided dataset files
├── docs/                       # Project notes and methodology docs
├── notebooks/                  # Exploratory analysis and modeling notebooks
├── outputs/
│   ├── models/                 # Saved trained models
│   └── recommendations/        # Generated recommendation outputs
├── reports/
│   └── figures/                # Plots for final report/slides
├── src/
│   ├── data/                   # Loading and cleaning code
│   ├── evaluation/             # Metrics and validation helpers
│   ├── features/               # Recipe/user feature engineering
│   ├── models/                 # SVD and Bayesian model code
│   └── recommender/            # End-to-end recommendation pipeline
├── tests/                      # Lightweight checks/tests
├── TODO.md                     # Project checklist
├── config.yaml                 # Paths and modeling defaults
└── requirements.txt            # Python dependencies
```

## Suggested Workflow

1. Run exploratory data analysis on recipes, ratings, sparsity, and outliers.
2. Clean ratings and recipe metadata.
3. Train an SVD collaborative filtering baseline.
4. Engineer interpretable recipe features from tags, ingredients, nutrition, and cooking time.
5. Build the Bayesian preference updater.
6. Combine SVD predictions with Bayesian posterior preferences.
7. Evaluate against validation/test ratings.
8. Generate final recommendations and report figures.

## Team Workflow

Use `WORK_SPLIT.md` to divide the work across three people:

- Person 1: data cleaning, EDA, and feature engineering
- Person 2: collaborative filtering baseline and SVD evaluation
- Person 3: Bayesian updating, hybrid scoring, and final recommendations

## Notes

Early data inspection found:

- Ratings are heavily skewed toward 5 stars.
- Ratings include `0`, which should be handled carefully.
- Recipe cooking times contain extreme outliers, including invalid huge values.
- The provided train/validation/test files already include internal user/item IDs as `u` and `i`.
