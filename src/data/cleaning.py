import pandas as pd


def filter_zero_ratings(interactions: pd.DataFrame) -> pd.DataFrame:
    """Remove zero ratings when treating the task as explicit 1-5 star prediction."""
    return interactions.loc[interactions["rating"] > 0].copy()


def cap_recipe_minutes(recipes: pd.DataFrame, max_minutes: int = 1440) -> pd.DataFrame:
    """Remove recipes with invalid or extreme cooking times."""
    cleaned = recipes.copy()
    cleaned = cleaned.loc[cleaned["minutes"].between(1, max_minutes)].copy()
    return cleaned

