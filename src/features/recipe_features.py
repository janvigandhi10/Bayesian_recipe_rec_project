import ast

import pandas as pd


DIETARY_FEATURES = {
    "is_vegetarian": "vegetarian",
    "is_vegan": "vegan",
    "is_gluten_free": "gluten-free",
    "is_low_carb": "low-carb",
    "is_low_fat": "low-fat",
    "is_healthy": "healthy",
}

MEAL_FEATURES = {
    "is_main_dish": "main-dish",
    "is_dessert": "desserts",
    "is_breakfast": "breakfast",
    "is_lunch": "lunch",
    "is_appetizer": "appetizers",
    "is_side_dish": "side-dishes",
    "is_snack": "snacks",
    "is_soup_stew": "soups-stews",
    "is_salad": "salads",
}

CUISINE_FEATURES = {
    "is_north_american": "north-american",
    "is_italian": "italian",
    "is_mexican": "mexican",
    "is_chinese": "chinese",
    "is_indian": "indian",
    "is_thai": "thai",
    "is_greek": "greek",
    "is_french": "french",
    "is_middle_eastern": "middle-eastern",
    "is_spanish": "spanish",
}

NUTRITION_COLS = [
    "calories",
    "total_fat_pct",
    "sugar_pct",
    "sodium_pct",
    "protein_pct",
    "saturated_fat_pct",
    "carbohydrates_pct",
]


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
    featured["parsed_ingredients"] = featured["ingredients"].apply(
        parse_list_column
    )

    featured["is_vegetarian"] = featured["parsed_tags"].apply(
        lambda tags: int("vegetarian" in tags)
    )

    featured["is_dessert"] = featured["parsed_tags"].apply(
        lambda tags: int("dessert" in tags or "desserts" in tags)
    )

    return featured


def parse_recipe_columns(recipes: pd.DataFrame) -> pd.DataFrame:
    """Parse tags, ingredients, and nutrition columns."""
    featured = recipes.copy()

    # Parse tags only if needed
    if "parsed_tags" not in featured.columns:
        featured["parsed_tags"] = featured["tags"].apply(parse_list_column)

    # Parse ingredients only if needed
    if "parsed_ingredients" not in featured.columns:
        featured["parsed_ingredients"] = featured["ingredients"].apply(
            parse_list_column
        )

    # Parse nutrition only if nutrition columns do not already exist
    missing_nutrition_cols = [
        col for col in NUTRITION_COLS
        if col not in featured.columns
    ]

    if missing_nutrition_cols:
        nutrition_values = featured["nutrition"].apply(parse_list_column)

        nutrition_df = pd.DataFrame(
            nutrition_values.tolist(),
            columns=NUTRITION_COLS,
            index=featured.index,
        )

        featured[NUTRITION_COLS] = nutrition_df

    return featured


def add_tag_features(
    recipes: pd.DataFrame,
    feature_map: dict,
) -> pd.DataFrame:
    """Create binary recipe features from parsed tags."""
    featured = recipes.copy()

    for column, tag in feature_map.items():
        featured[column] = featured["parsed_tags"].apply(
            lambda tags: int(tag in tags)
        )

    return featured


def add_time_features(recipes: pd.DataFrame) -> pd.DataFrame:
    """Create mutually exclusive cooking-time features."""
    featured = recipes.copy()

    featured["time_bucket"] = pd.cut(
        featured["minutes"],
        bins=[0, 30, 60, 120, float("inf")],
        labels=["quick", "moderate", "long", "extended"],
        include_lowest=True,
    )

    featured["is_quick"] = (
        featured["time_bucket"] == "quick"
    ).astype(int)

    featured["is_moderate_time"] = (
        featured["time_bucket"] == "moderate"
    ).astype(int)

    featured["is_long_time"] = (
        featured["time_bucket"] == "long"
    ).astype(int)

    featured["is_extended_time"] = (
        featured["time_bucket"] == "extended"
    ).astype(int)

    return featured


def add_nutrition_features(recipes: pd.DataFrame) -> pd.DataFrame:
    """Create low/high nutrition indicators using quartile cutoffs."""
    featured = recipes.copy()

    for col in NUTRITION_COLS:
        q25 = featured[col].quantile(0.25)
        q75 = featured[col].quantile(0.75)

        level_col = f"{col}_level"

        featured[level_col] = pd.cut(
            featured[col],
            bins=[-float("inf"), q25, q75, float("inf")],
            labels=["low", "moderate", "high"],
            include_lowest=True,
        )

        feature_name = col.replace("_pct", "")

        featured[f"is_low_{feature_name}"] = (
            featured[level_col] == "low"
        ).astype(int)

        featured[f"is_high_{feature_name}"] = (
            featured[level_col] == "high"
        ).astype(int)

    return featured


def build_recipe_features(recipes: pd.DataFrame) -> pd.DataFrame:
    """Build the final feature-ready recipe dataset."""
    featured = parse_recipe_columns(recipes)

    featured = add_tag_features(featured, DIETARY_FEATURES)
    featured = add_tag_features(featured, MEAL_FEATURES)
    featured = add_tag_features(featured, CUISINE_FEATURES)
    featured = add_time_features(featured)
    featured = add_nutrition_features(featured)

    nutrition_feature_cols = []
    for col in NUTRITION_COLS:
        feature_name = col.replace("_pct", "")
        nutrition_feature_cols.extend(
            [
                f"is_low_{feature_name}",
                f"is_high_{feature_name}",
            ]
        )

    final_feature_cols = (
        list(DIETARY_FEATURES.keys())
        + list(MEAL_FEATURES.keys())
        + list(CUISINE_FEATURES.keys())
        + [
            "is_quick",
            "is_moderate_time",
            "is_long_time",
            "is_extended_time",
        ]
        + nutrition_feature_cols
    )

    return featured[
        ["id", "name"] + final_feature_cols
    ].copy()