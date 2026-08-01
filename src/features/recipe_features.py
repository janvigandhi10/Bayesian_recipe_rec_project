import ast

import pandas as pd


def parse_list_column(value):
    """Parse Food.com list-like string columns into Python lists."""
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    return parsed if isinstance(parsed, list) else []


def add_basic_recipe_features(recipes: pd.DataFrame) -> pd.DataFrame:
    """Add simple feature columns useful for exploration and modeling."""
    featured = recipes.copy()
    featured["parsed_tags"] = featured["tags"].apply(parse_list_column)
    featured["parsed_ingredients"] = featured["ingredients"].apply(parse_list_column)
    featured["is_vegetarian"] = featured["parsed_tags"].apply(lambda tags: "vegetarian" in tags)
    featured["is_dessert"] = featured["parsed_tags"].apply(lambda tags: "dessert" in tags or "desserts" in tags)
    return featured

