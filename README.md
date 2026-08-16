# Bayesian Personalized Recipe Recommendation System

This project is a recipe recommendation system that combines **SVD
collaborative filtering** with **Bayesian updating** to recommend recipes a user
is likely to enjoy.

The final project includes:

- a cleaned notebook pipeline
- saved model outputs and evaluation metrics
- a customer-friendly Streamlit demo
- a simple Model Report section for explaining results in class

## Project Summary

Food recommendation systems often give recommendations without explaining why
the recipe was chosen or how confident the system is.

Our project tries to solve that by building a hybrid recommender that uses:

- the user's recipe history
- similarities to other users
- cuisine preferences
- dietary needs
- ingredients
- cooking time
- historical recipe ratings
- quick feedback such as **Like this**, **Not for me**, and **Skip**

The goal is not just to recommend a recipe. The goal is to recommend a recipe
and explain why it fits the user.

## Main Idea

Our recommender has two main parts:

1. **SVD** learns from historical ratings.
2. **Bayesian updating** adjusts recommendations as the user gives feedback.

Key takeaway:

*SVD tells us what similar users liked, and Bayesian updating tells us how to
adjust that recommendation for this specific user's current preferences.*

## What Is SVD?

SVD stands for **Singular Value Decomposition**. In this project, we use SVD as a
collaborative filtering model.

That means SVD looks at the Food.com rating history and learns patterns like:

- users who liked recipe A also liked recipe B
- users with similar taste often rate similar recipes highly
- some recipes are generally rated higher than others

Simple example:

If a user liked **Butter Chicken** and **Garlic Naan**, SVD can look at other
users with similar taste and suggest recipes those users also liked, such as
**Chicken Tikka Masala** or **Paneer Butter Masala**.

What SVD answers:

*Based on past ratings from many users, what recipes is this user likely to
enjoy?*

In the Streamlit app, SVD is used in **Saved Customer Profile** mode because
those users already have rating history.

## What Is Bayesian Updating?

Bayesian updating starts with an initial belief, called a **prior**, and updates
that belief when new evidence comes in.

In our project:

- the prior comes from historical recipe and user data
- new evidence comes from user feedback
- the updated belief is called the posterior

Simple example:

If a user says they like Mexican main dishes, the app increases the user's
preference for recipes with Mexican and main-dish tags.

If the user says **Not for me** on a dessert, dessert-like recipes become less
important for that user.

If the user clicks **Skip**, the app moves on without treating that recipe as
positive or negative evidence.

What Bayesian updating answers:

*Given this specific user's recent choices, how should we adjust the
recommendation, and how confident are we?*

This is useful for new users because they may not have rating history yet.

## Why Use Both?

SVD and Bayesian updating solve different problems.

| Model part | What it is good at |
| --- | --- |
| SVD | Learning from many users and past ratings |
| Bayesian updating | Adapting quickly to one user's feedback |
| Hybrid model | Combining both signals into one recommendation |

The hybrid model is useful because:

- SVD captures patterns across similar users
- Bayesian updating captures the current user's preferences
- recipe quality priors help avoid weak recommendations
- confidence scores make the recommendation easier to explain

## Hybrid Recommendation Score

For saved customers, the app combines SVD and Bayesian/content information:

```text
final score = 70% * SVD score + 30% * Bayesian/content score
predicted stars = 1 + 4 * final score
```

The 70/30 weight comes from the hybrid notebook.

In the notebook, we tested different ways to blend the SVD score and the
Bayesian/content score. The best validation blend for rating prediction used
more weight on SVD and less weight on Bayesian/content:

- **70% SVD** because saved users have rating history, so collaborative
  filtering is very useful.
- **30% Bayesian/content** because recipe tags, ingredients, quality priors, and
  user preferences still add useful personalization and explanation.

This weight is fixed in the app. The app does not randomly choose a new model
every time.

For quick-quiz users, the app uses Bayesian updating and recipe quality because
a brand-new user does not have an SVD history yet.

## Data

The project uses the Food.com Recipes and User Interactions dataset from Kaggle.

The dataset includes:

- over 180,000 recipes
- about 700,000 user ratings and reviews
- ingredients
- nutrition information
- cooking instructions
- recipe tags such as vegetarian, dessert, Italian, gluten-free, and main dish
- historical user-recipe interactions

Key files are stored in `Data/`:

- `RAW_recipes.csv`
- `RAW_interactions.csv`
- `interactions_train.csv`
- `interactions_validation.csv`
- `interactions_test.csv`
- `PP_recipes.csv`
- `PP_users.csv`
- `ingr_map.pkl`

## EDA And Feature Engineering

Before modeling, we explored the Food.com data to understand what the model
would be learning from.

Important EDA findings:

- The dataset is large, with many recipes and many user ratings.
- Ratings are very positive overall, with many 5-star ratings.
- Some ratings are `0`, so we handled those carefully because they do not work
  like normal 1-5 star ratings.
- Cooking times have outliers, including extremely large or invalid values.
- Recipes include useful metadata such as tags, ingredients, nutrition, minutes,
  descriptions, and cooking steps.

### EDA Visuals

The first useful visual is the rating distribution:

![Distribution of explicit recipe ratings](reports/figures/rating_distribution.png)

Takeaway:

*Most ratings are 5 stars, so the model has to work with a very positive and
imbalanced rating dataset.*

The second useful visual is average recipe rating compared with number of
ratings:

![Average recipe rating vs number of ratings](reports/figures/recipe_rating_vs_count.png)

Takeaway:

*Some recipes have very high average ratings but only a few reviews. This is why
we use recipe quality priors and confidence, instead of trusting every average
rating equally.*

We then turned the raw recipe data into features the model could use.

Feature groups we created:

| Feature group | Examples |
| --- | --- |
| Cuisine | Italian, Mexican, Indian, Chinese, Thai, Greek, French |
| Dietary | Vegetarian, vegan, gluten-free, low-carb, healthy |
| Dish type | Main dish, dessert, breakfast, appetizer, salad, soup |
| Time | Quick, moderate, long, extended |
| Nutrition | Calories and nutrition buckets |
| Ingredients | Chicken, beef, cheese, chocolate, garlic, mushroom, potato, shrimp |

Key takeaway:

*The feature engineering step turns recipe information like tags, ingredients,
nutrition, and cooking time into signals the recommender can learn from.*

## Notebook Pipeline

The final notebooks are in `notebooks/final/`.

Present them in this order:

1. `01_data_exploration.ipynb`
2. `02_svd_baseline.ipynb`
3. `03_bayesian_updating.ipynb`
4. `04_hybrid_recommender.ipynb`

What each notebook does:

| Notebook | Purpose |
| --- | --- |
| Data exploration | Understand recipes, ratings, missing values, and rating patterns |
| SVD baseline | Train a collaborative filtering model using Surprise SVD |
| Bayesian updating | Build user preference posteriors and recipe quality priors |
| Hybrid recommender | Combine SVD and Bayesian scores and evaluate the final system |

## Streamlit Demo

Run the demo with:

```powershell
$env:PYTHONPATH='.'
python -m streamlit run streamlit_app.py
```

Then open:

```text
http://localhost:8501
```

The app has three tabs:

| Tab | What it does |
| --- | --- |
| Find Recipes | Live customer-facing recommendation demo |
| Model Report | Fixed notebook results explained in plain English |
| About | Short summary of how the demo works |

## How The Demo Works

### Quick Taste Quiz

This mode is for a brand-new user.

The user can choose:

- cuisine
- dish type
- dietary needs
- maximum cooking time
- ingredients they have or want

Then the user gives fast feedback:

- **Like this** means positive evidence
- **Not for me** means negative evidence
- **Skip** means no evidence

The app uses this feedback to update a temporary Bayesian taste profile and
recommend better recipes.

Important detail:

- The Bayesian part updates from the user's feedback.
- Each **Like this** or **Not for me** click becomes new evidence for the next
  recommendations.
- **Skip** does not change the user's taste profile.
- The SVD model does not retrain after each click. It is already trained from
  historical ratings and gives a stable collaborative filtering signal.

Simple explanation:

*The app updates the Bayesian preference profile live, but SVD stays fixed as
the historical model.*

### Saved Customer Profile

This mode is for a user who already exists in the rating data.

Because the user has history, the app can use:

- SVD predictions
- Bayesian user preferences
- recipe quality priors
- confidence scores

This shows how the full hybrid system works.

## Class Demo Walkthrough

Start with the **Find Recipes** tab.

1. Choose **Quick Taste Quiz**.
2. Select a cuisine, dish type, dietary need, and cooking time.
3. Answer the quick taste questions.
4. Review the top picks one by one.
5. Click **Like this**, **Not for me**, or **Skip**.
6. Show that liked recipes appear in **Your Selection**.
7. Choose a recipe from the selection table.
8. Click **Cook selected recipe**.
9. Show the full recipe description, ingredients, and steps.

Then switch to **Saved Customer Profile**.

Takeaway:

*In quick quiz mode, the user is new, so the app uses Bayesian feedback. In
saved customer mode, the user has rating history, so the app can also use SVD.*

Then open the **Model Report** tab.

Takeaway:

*This tab is the model's fixed report card from the notebooks. These numbers do
not change during the demo because they come from saved evaluation datasets.*

## Warm vs. Cold Test Sets

The Model Report uses two fixed evaluation sets.

They are not affected by what the user clicks in the app.

| Test set | Simple meaning |
| --- | --- |
| Warm holdout | Easier case where the model has history for similar users or recipes |
| Cold test | Harder case where the model has less history for the recipe |

Why we use both:

- Warm holdout shows how the model works for familiar users and recipes.
- Cold test shows whether the Bayesian/content part helps when there is less
  rating history.

Key takeaway:

*Warm means familiar. Cold means harder because there is less history.*

## Main Results

Lower RMSE means better rating prediction.

| Result | Takeaway |
| --- | --- |
| Warm holdout | SVD performs best for familiar users and recipes, with RMSE about `0.694`. |
| Cold test | The hybrid model improves over SVD, with hybrid RMSE about `0.855` compared with SVD RMSE about `0.886`. |
| Ranking at 10 | Popularity is still strong, so our demo focuses on personalization, explanations, and confidence. |

This does not mean SVD is always better than Bayesian.

The clearer interpretation is:

- SVD is strongest when the user already has rating history.
- Bayesian updating is useful when we need quick personalization, explanations,
  confidence, or support for newer recipes with less history.
- The hybrid approach is the main project idea because it combines SVD's
  historical rating patterns with Bayesian adaptability.

Main conclusion:

*SVD is strong when there is user history. Bayesian updating is useful for
quick personalization, uncertainty, and newer recipes. The hybrid system brings
both together.*

Broader conclusion:

*The best recommendation system is not only the one with the lowest error. For
a real user, it also needs to adapt quickly, explain its choices, respect
dietary and time constraints, and show confidence. Our hybrid approach is
useful because it connects model performance with a more understandable user
experience.*

## Important Saved Outputs

The app reads these saved files:

- `outputs/hannah_recipe_quality.csv`
- `outputs/hannah_user_posteriors.csv`
- `outputs/hannah_user_tag_posteriors.csv`
- `outputs/hannah_hybrid_metrics.csv`
- `outputs/hannah_ranking_metrics_hybrid.csv`
- `outputs/hannah_calibration.csv`
- `outputs/hannah_bayes_params.json`
- `outputs/models/hannah_svd.pkl`

The SVD model file is stored locally in `outputs/models/`. It is ignored by git
because it is large. If it is missing, rerun the SVD notebook.

## Repository Structure

```text
Project_bayesian/
|-- Data/                 # Food.com data
|-- docs/                 # Project notes
|-- notebooks/
|   |-- final/            # Final notebook pipeline
|   `-- supporting/       # Supporting teammate notebooks
|-- outputs/              # Saved model outputs and metrics
|-- reports/              # Supporting figures and results
|-- src/                  # Reusable code
|-- tests/                # Tests
|-- streamlit_app.py      # Streamlit demo
|-- README.md             # Presentation README
|-- WORK_SPLIT.md         # Team split
`-- requirements.txt      # Dependencies
```

## Team Contributions

- **Hannah's work**: final modeling spine, Bayesian artifacts, hybrid metrics,
  user/tag posterior outputs, recipe quality artifacts.
- **Hoang's work**: supporting EDA and SVD baseline exploration.
- **Janvi's work**: project abstract and methodology, initial scaffolding and
  repository structure, final integration, and Streamlit demo.

## How To Verify

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q
```

Current status:

```text
32 passed
```

## Final Presentation Summary

Short version:

*We built a hybrid recipe recommender using Food.com data. SVD learns from
similar users and past ratings. Bayesian updating adapts to a user's current
feedback and gives interpretable confidence. The Streamlit app turns this into
a customer-facing demo where users can filter recipes, give quick feedback,
save liked options, and view full cooking steps.*
