# Data

Place the Food.com Recipes and User Interactions dataset files in this folder.

## Raw Dataset Files

Expected local files:

- `RAW_recipes.csv`
- `RAW_interactions.csv`
- `interactions_train.csv`
- `interactions_validation.csv`
- `interactions_test.csv`
- `PP_recipes.csv`
- `PP_users.csv`
- `ingr_map.pkl`

## Processed Data

The data cleaning and feature-engineering pipeline creates the following files locally in `Data/processed/`:

- `recipe_features.csv`
  - 231,637 recipes
  - Recipe ID and name
  - 43 engineered features covering dietary preferences, meal categories, cuisines, cooking-time categories, and nutrition characteristics

- `explicit_ratings.csv`
  - 1,071,520 user-recipe interactions
  - Contains explicit ratings from 1 to 5
  - Includes `user_id`, `recipe_id`, `date`, `rating`, and `review`

These files can be reproduced using the cleaning and feature-engineering code in:

- `src/data/cleaning.py`
- `src/features/recipe_features.py`

## Cleaning Decisions

The following preprocessing decisions are used for downstream modeling:

- Rating `0` is excluded from explicit-rating modeling because inspection showed that zero-rated interactions contain written reviews and do not consistently represent negative feedback.
- Explicit ratings are therefore restricted to the original 1-5 rating scale.
- Recipe names with missing values are assigned `"Unknown Recipe"`.
- Recipes with zero-minute preparation times are treated as having unavailable cooking-time information rather than as true zero-minute recipes.
- The extreme preparation-time value `2,147,483,647` minutes is treated as invalid and replaced with missing cooking-time information.
- Other large cooking-time values are retained because some represent legitimate long-duration recipes such as aging, fermenting, or preserving.

## Version Control

The large raw and processed dataset files are intentionally ignored by Git so the repository stays lightweight. Only this README and the code required to reproduce the processed datasets are tracked.

## Dataset Source

[Food.com Recipes and User Interactions - Kaggle](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)