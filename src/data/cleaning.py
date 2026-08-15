import pandas as pd


def filter_zero_ratings(interactions: pd.DataFrame) -> pd.DataFrame:
    """Remove zero ratings when treating the task as explicit 1-5 star prediction."""
    return interactions.loc[interactions["rating"] > 0].copy()


def cap_recipe_minutes(
    recipes: pd.DataFrame,
    max_minutes: int = 1440,
) -> pd.DataFrame:
    """Remove recipes with invalid or extreme cooking times."""
    cleaned = recipes.copy()
    cleaned = cleaned.loc[
        cleaned["minutes"].between(1, max_minutes)
    ].copy()
    return cleaned


def clean_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    """Prepare interactions for explicit-rating modeling."""
    cleaned = interactions.copy()

    # Keep only explicit 1-5 ratings
    cleaned = cleaned.loc[
        cleaned["rating"].between(1, 5)
    ].copy()

    # Convert date to datetime
    if "date" in cleaned.columns:
        cleaned["date"] = pd.to_datetime(
            cleaned["date"],
            errors="coerce",
        )

    return cleaned


def clean_recipes(recipes: pd.DataFrame) -> pd.DataFrame:
    """Apply finalized recipe cleaning decisions."""
    cleaned = recipes.copy()

    # Preserve recipes with missing names using a placeholder
    if "name" in cleaned.columns:
        cleaned["name"] = cleaned["name"].fillna(
            "Unknown Recipe"
        )

    # Treat invalid preparation times as unavailable
    if "minutes" in cleaned.columns:
        invalid_minutes = (
            (cleaned["minutes"] <= 0)
            | (cleaned["minutes"] == 2147483647)
        )

        cleaned.loc[invalid_minutes, "minutes"] = pd.NA
    return cleaned