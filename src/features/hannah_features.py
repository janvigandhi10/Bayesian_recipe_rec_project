"""Shared feature-family definitions for the hannah_* notebooks.

Extracted verbatim-in-logic from the feature-family cell of
``notebooks/final/03_bayesian_updating.ipynb`` (which was semantically
identical to the copy in ``final/04_hybrid_recommender.ipynb``), so both notebooks import a single
source of truth. Bump ``FEATURE_VERSION`` whenever CANDIDATES /
ING_KEYWORDS / flag construction change; the notebooks print it as a drift
guard.
"""

from __future__ import annotations

import pandas as pd

from .recipe_features import parse_list_column

FEATURE_VERSION = "2026-08-14-v2"

# Candidate tag families (family -> specific Food.com tags), as in hannah_03.
CANDIDATES = {
    "cuisine": ["italian", "mexican", "asian", "american", "french", "greek", "indian",
                "chinese", "thai", "middle-eastern"],
    "dietary": ["vegetarian", "vegan", "gluten-free", "low-carb", "low-fat", "low-sodium",
                "low-cholesterol", "healthy"],
    "dish":    ["desserts", "main-dish", "side-dishes", "salads", "soups-stews",
                "breakfast", "appetizers", "breads"],
    "time":    ["15-minutes-or-less", "30-minutes-or-less", "60-minutes-or-less",
                "4-hours-or-less"],
}
CAND_TAGS = [t for tags in CANDIDATES.values() for t in tags]

# Ingredient family: substring match on each parsed ingredient string catches
# variants ("boneless skinless chicken breasts" -> chicken).
ING_KEYWORDS = ["chicken", "beef", "cheese", "chocolate", "garlic", "mushroom",
                "potato", "shrimp"]

# Calorie terciles form the specific members of the nutrition family.
CAL_TERCILE_TAGS = ["cal_low", "cal_mid", "cal_high"]

FAMILIES = ["cuisine", "dietary", "dish", "time", "nutrition", "ingredient"]
FAMILY_COLS = [f"any_{fam}" for fam in FAMILIES]

# family -> specific tag names as used in per-tag posterior tables
# (CANDIDATES members + calorie terciles + ingredient keywords).
TAG_MEMBERS = {**{fam: list(tags) for fam, tags in CANDIDATES.items()},
               "nutrition": list(CAL_TERCILE_TAGS),
               "ingredient": list(ING_KEYWORDS)}


def flag_col(family: str, tag: str) -> str:
    """Column name in build_recipe_flags() output for a specific (family, tag)."""
    return f"tag__{family}__{tag}"


def tag_flag_columns() -> list:
    """All specific-tag flag column names, family by family (contract order)."""
    return [flag_col(fam, tag) for fam in FAMILIES for tag in TAG_MEMBERS[fam]]


def parse_tags(recipes_df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``recipes_df`` with a ``tag_set`` column of parsed tag sets.

    Uses :func:`src.features.recipe_features.parse_list_column`, so malformed /
    missing ``tags`` values yield an empty set rather than raising.
    """
    out = recipes_df.copy()
    out["tag_set"] = out["tags"].map(parse_list_column).map(set)
    return out


def parse_calories(recipes_df: pd.DataFrame) -> pd.Series:
    """Calories per recipe: first entry of the ``nutrition`` list, NaN if unparseable."""
    return pd.to_numeric(
        recipes_df["nutrition"].astype(str).str.strip("[]").str.split(",", n=1).str[0],
        errors="coerce",
    )


def calorie_terciles(recipes_df: pd.DataFrame) -> tuple:
    """(P33, P67) of cleaned calories within ``recipes_df`` (the tercile cuts)."""
    calories = parse_calories(recipes_df)
    p33, p67 = calories.quantile([1 / 3, 2 / 3])
    return float(p33), float(p67)


def build_recipe_flags(recipes_df: pd.DataFrame) -> pd.DataFrame:
    """Boolean feature flags per recipe, indexed by the RAW_recipes ``id`` column.

    Columns:
    - one bool per specific candidate tag: ``tag__<family>__<tag>``
    - calorie terciles: ``tag__nutrition__cal_low/cal_mid/cal_high``
      (tercile cuts computed on cleaned calories *within the passed frame*;
      rows with unparseable calories get all three flags False)
    - one bool per ingredient keyword: ``tag__ingredient__<kw>``
    - the six aggregates ``any_<family>``; ``any_nutrition`` reproduces the
      notebooks' original logic (low OR high calorie, i.e. mid-tercile rows
      are *not* counted as nutrition-flagged).
    """
    tag_sets = recipes_df["tags"].map(parse_list_column).map(set)
    flags = pd.DataFrame(index=pd.Index(recipes_df["id"].to_numpy(), name="id"))

    for fam, fam_tags in CANDIDATES.items():
        for t in fam_tags:
            flags[flag_col(fam, t)] = tag_sets.map(lambda s, t=t: t in s).to_numpy()
        fam_cols = [flag_col(fam, t) for t in fam_tags]
        flags[f"any_{fam}"] = flags[fam_cols].any(axis=1)

    # nutrition family: calories is the first entry of the nutrition list
    calories = parse_calories(recipes_df)
    p33, p67 = calories.quantile([1 / 3, 2 / 3])
    cal_low = (calories <= p33).to_numpy()
    cal_high = (calories >= p67).to_numpy()
    cal_mid = calories.notna().to_numpy() & ~cal_low & ~cal_high
    flags[flag_col("nutrition", "cal_low")] = cal_low
    flags[flag_col("nutrition", "cal_mid")] = cal_mid
    flags[flag_col("nutrition", "cal_high")] = cal_high
    # original notebook logic: any_nutrition = low_calorie | high_calorie
    flags["any_nutrition"] = cal_low | cal_high

    # ingredient family: membership tests on the parsed ingredients list
    ing_joined = recipes_df["ingredients"].map(parse_list_column).map(" | ".join)
    for kw in ING_KEYWORDS:
        flags[flag_col("ingredient", kw)] = ing_joined.str.contains(kw, regex=False).to_numpy()
    ing_cols = [flag_col("ingredient", kw) for kw in ING_KEYWORDS]
    flags["any_ingredient"] = flags[ing_cols].any(axis=1)

    return flags
