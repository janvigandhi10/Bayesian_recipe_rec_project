import pandas as pd
import pytest

from src.features.hannah_features import (
    CANDIDATES,
    CAL_TERCILE_TAGS,
    FAMILIES,
    FAMILY_COLS,
    FEATURE_VERSION,
    ING_KEYWORDS,
    TAG_MEMBERS,
    build_recipe_flags,
    calorie_terciles,
    flag_col,
    parse_calories,
    parse_tags,
    tag_flag_columns,
)


@pytest.fixture
def toy_recipes():
    return pd.DataFrame({
        "id": [101, 202, 303],
        "tags": [
            "['italian', 'vegetarian', 'main-dish', '30-minutes-or-less']",
            "['thai', 'desserts']",
            "['weeknight']",
        ],
        "nutrition": [
            "[100.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]",
            "[500.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]",
            "[900.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]",
        ],
        "ingredients": [
            "['boneless skinless chicken breasts', 'garlic cloves']",
            "['dark chocolate', 'butter']",
            "['flour', 'water']",
        ],
    })


def test_module_constants():
    assert FEATURE_VERSION == "2026-08-14-v2"
    assert FAMILIES == ["cuisine", "dietary", "dish", "time", "nutrition", "ingredient"]
    assert FAMILY_COLS == [f"any_{f}" for f in FAMILIES]
    assert set(CANDIDATES) == {"cuisine", "dietary", "dish", "time"}
    assert len(ING_KEYWORDS) == 8
    assert TAG_MEMBERS["nutrition"] == CAL_TERCILE_TAGS
    assert TAG_MEMBERS["ingredient"] == ING_KEYWORDS
    assert TAG_MEMBERS["cuisine"] == CANDIDATES["cuisine"]


def test_parse_tags_returns_sets(toy_recipes):
    parsed = parse_tags(toy_recipes)
    assert "tag_set" in parsed.columns
    assert parsed.loc[0, "tag_set"] == {"italian", "vegetarian", "main-dish",
                                        "30-minutes-or-less"}
    # does not mutate the input
    assert "tag_set" not in toy_recipes.columns


def test_parse_tags_malformed_yields_empty_set():
    df = pd.DataFrame({"id": [1], "tags": ["not a list"]})
    assert parse_tags(df).loc[0, "tag_set"] == set()


def test_parse_calories_and_terciles(toy_recipes):
    cal = parse_calories(toy_recipes)
    assert cal.tolist() == [100.0, 500.0, 900.0]
    p33, p67 = calorie_terciles(toy_recipes)
    assert p33 == pytest.approx(1100 / 3, abs=1e-6)   # linear interp of [100,500,900]
    assert p67 == pytest.approx(1900 / 3, abs=1e-6)


def test_build_recipe_flags_index_and_columns(toy_recipes):
    flags = build_recipe_flags(toy_recipes)
    assert flags.index.name == "id"
    assert flags.index.tolist() == [101, 202, 303]
    # 30 candidate tags + 3 terciles + 8 ingredient kws + 6 any_* aggregates
    expected_cols = set(tag_flag_columns()) | set(FAMILY_COLS)
    assert set(flags.columns) == expected_cols
    assert len(flags.columns) == 30 + 3 + 8 + 6
    assert all(flags[c].dtype == bool for c in flags.columns)


def test_specific_tag_flags(toy_recipes):
    flags = build_recipe_flags(toy_recipes)
    assert flags.loc[101, flag_col("cuisine", "italian")]
    assert flags.loc[202, flag_col("cuisine", "thai")]
    assert not flags.loc[303, flag_col("cuisine", "italian")]
    assert flags.loc[101, flag_col("dietary", "vegetarian")]
    assert flags.loc[101, flag_col("dish", "main-dish")]
    assert flags.loc[202, flag_col("dish", "desserts")]
    assert flags.loc[101, flag_col("time", "30-minutes-or-less")]


def test_any_family_aggregates(toy_recipes):
    flags = build_recipe_flags(toy_recipes)
    assert flags["any_cuisine"].tolist() == [True, True, False]
    assert flags["any_dietary"].tolist() == [True, False, False]
    assert flags["any_dish"].tolist() == [True, True, False]
    assert flags["any_time"].tolist() == [True, False, False]


def test_calorie_tercile_flags(toy_recipes):
    flags = build_recipe_flags(toy_recipes)
    assert flags[flag_col("nutrition", "cal_low")].tolist() == [True, False, False]
    assert flags[flag_col("nutrition", "cal_mid")].tolist() == [False, True, False]
    assert flags[flag_col("nutrition", "cal_high")].tolist() == [False, False, True]
    # exactly one tercile per (parseable) row
    tercile_cols = [flag_col("nutrition", t) for t in CAL_TERCILE_TAGS]
    assert (flags[tercile_cols].sum(axis=1) == 1).all()
    # original notebook logic: any_nutrition = low | high (mid excluded)
    assert flags["any_nutrition"].tolist() == [True, False, True]


def test_ingredient_flags_substring_match(toy_recipes):
    flags = build_recipe_flags(toy_recipes)
    # "boneless skinless chicken breasts" -> chicken via substring match
    assert flags.loc[101, flag_col("ingredient", "chicken")]
    assert flags.loc[101, flag_col("ingredient", "garlic")]
    assert flags.loc[202, flag_col("ingredient", "chocolate")]
    assert not flags.loc[303, flag_col("ingredient", "chicken")]
    assert flags["any_ingredient"].tolist() == [True, True, False]


def test_unparseable_calories_get_no_tercile_flags():
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "tags": ["['italian']", "[]", "[]"],
        "nutrition": ["[100.0, 1.0]", "[300.0, 1.0]", None],
        "ingredients": ["['salt']", "['salt']", "['salt']"],
    })
    flags = build_recipe_flags(df)
    tercile_cols = [flag_col("nutrition", t) for t in CAL_TERCILE_TAGS]
    assert flags.loc[3, tercile_cols].tolist() == [False, False, False]
    assert not flags.loc[3, "any_nutrition"]
