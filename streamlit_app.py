from __future__ import annotations

import json
import os
import pickle
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.features.hannah_features import FAMILIES, TAG_MEMBERS, build_recipe_flags, flag_col
from src.features.recipe_features import parse_list_column
from src.models.bayesian_updater import ci_width, posterior_mean


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Data"
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
SURPRISE_CACHE_DIR = ROOT / ".surprise_data"
os.environ.setdefault("SURPRISE_DATA_FOLDER", str(SURPRISE_CACHE_DIR))

FAMILY_WEIGHT = 0.25
TAG_WEIGHT = 0.35
QUALITY_WEIGHT = 0.25
BASE_WEIGHT = 1.0 - FAMILY_WEIGHT - TAG_WEIGHT - QUALITY_WEIGHT
SVD_HYBRID_WEIGHT = 0.70

DIETARY_FILTERS = {
    "Vegetarian": flag_col("dietary", "vegetarian"),
    "Vegan": flag_col("dietary", "vegan"),
    "Gluten-free": flag_col("dietary", "gluten-free"),
    "Low-carb": flag_col("dietary", "low-carb"),
    "Healthy": flag_col("dietary", "healthy"),
}

PANTRY_FILTERS = {
    "Chicken": flag_col("ingredient", "chicken"),
    "Beef": flag_col("ingredient", "beef"),
    "Cheese": flag_col("ingredient", "cheese"),
    "Chocolate": flag_col("ingredient", "chocolate"),
    "Garlic": flag_col("ingredient", "garlic"),
    "Mushroom": flag_col("ingredient", "mushroom"),
    "Potato": flag_col("ingredient", "potato"),
    "Shrimp": flag_col("ingredient", "shrimp"),
}

QUICK_TAG_CHOICES = {
    "Cuisines": {
        "Italian": ("cuisine", "italian"),
        "Mexican": ("cuisine", "mexican"),
        "Asian": ("cuisine", "asian"),
        "Indian": ("cuisine", "indian"),
        "Thai": ("cuisine", "thai"),
        "Greek": ("cuisine", "greek"),
    },
    "Meal types": {
        "Desserts": ("dish", "desserts"),
        "Main dishes": ("dish", "main-dish"),
        "Breakfast": ("dish", "breakfast"),
        "Salads": ("dish", "salads"),
        "Soups": ("dish", "soups-stews"),
        "Appetizers": ("dish", "appetizers"),
    },
    "Ingredients": {
        label: ("ingredient", label.lower())
        for label in PANTRY_FILTERS
    },
}

RATING_OPTIONS = {
    "Skip": None,
    "Love it": True,
    "Not for me": False,
}

QUICK_CONCEPTS = [
    {
        "id": "quesadillas",
        "name": "Quesadillas",
        "description": "Melty cheese, tortillas, salsa, and Mexican-inspired flavors.",
        "preferences": [("cuisine", "mexican"), ("ingredient", "cheese"), ("dish", "main-dish")],
    },
    {
        "id": "curry",
        "name": "Curries",
        "description": "Warm spices, saucy main dishes, and Indian or Thai-style meals.",
        "preferences": [("cuisine", "indian"), ("cuisine", "thai"), ("dish", "main-dish")],
    },
    {
        "id": "fondue_dips",
        "name": "Fondue and dips",
        "description": "Cheesy, creamy, snackable dishes for sharing.",
        "preferences": [("dish", "appetizers"), ("ingredient", "cheese")],
    },
    {
        "id": "pasta_bakes",
        "name": "Pasta bakes",
        "description": "Comforting Italian-style main dishes with sauce and cheese.",
        "preferences": [("cuisine", "italian"), ("dish", "main-dish"), ("ingredient", "cheese")],
    },
    {
        "id": "fresh_salads",
        "name": "Fresh salads",
        "description": "Lighter meals with greens, vegetables, and bright flavors.",
        "preferences": [("dish", "salads"), ("dietary", "healthy")],
    },
    {
        "id": "chocolate_desserts",
        "name": "Chocolate desserts",
        "description": "Sweet recipes built around chocolate and dessert flavors.",
        "preferences": [("dish", "desserts"), ("ingredient", "chocolate")],
    },
    {
        "id": "garlic_chicken",
        "name": "Garlic chicken",
        "description": "Savory chicken dishes with garlic-forward seasoning.",
        "preferences": [("ingredient", "chicken"), ("ingredient", "garlic"), ("dish", "main-dish")],
    },
    {
        "id": "shrimp_stir_fry",
        "name": "Shrimp stir-fry",
        "description": "Fast seafood meals with Asian-inspired flavors.",
        "preferences": [("ingredient", "shrimp"), ("cuisine", "asian"), ("dish", "main-dish")],
    },
]

QUICK_CONCEPT_MAP = {concept["id"]: concept for concept in QUICK_CONCEPTS}

INCOMPATIBLE_DISH_TAGS = {
    "main-dish": ["desserts", "breakfast", "appetizers", "breads", "salads", "side-dishes"],
    "desserts": ["main-dish", "breakfast", "salads", "soups-stews"],
    "breakfast": ["main-dish", "desserts", "soups-stews"],
}


st.set_page_config(
    page_title="MealMatch",
    page_icon="",
    layout="wide",
)


st.markdown(
    """
<style>
    :root {
        --meal-bg: #f6f3ea;
        --meal-panel: #ffffff;
        --meal-panel-soft: #fffaf0;
        --meal-border: #d9d2c2;
        --meal-text: #1f2933;
        --meal-muted: #5f6b7a;
        --meal-green: #2f6b4f;
        --meal-gold: #b7791f;
        --meal-red: #9b2c2c;
    }
    .stApp {
        background: var(--meal-bg);
        color: var(--meal-text);
    }
    .block-container {
        padding-top: 2rem;
        max-width: 1180px;
    }
    section[data-testid="stSidebar"] {
        background: #efe7d7;
        border-right: 1px solid var(--meal-border);
    }
    .main h1,
    .main h2,
    .main h3,
    .main p,
    .main label {
        color: var(--meal-text);
    }
    .app-kicker {
        color: var(--meal-green);
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }
    .app-title {
        font-size: 2.35rem;
        font-weight: 750;
        line-height: 1.1;
        margin-bottom: 0.4rem;
        color: var(--meal-text);
    }
    .app-subtitle {
        color: var(--meal-muted);
        font-size: 1.05rem;
        max-width: 820px;
        margin-bottom: 1.25rem;
    }
    div[data-testid="stTabs"] button {
        color: inherit;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--meal-green);
        border-bottom-color: var(--meal-green);
    }
    .recipe-card {
        border: 1px solid var(--meal-border);
        border-radius: 8px;
        padding: 1rem 1.05rem;
        margin-bottom: 0.8rem;
        background: #fffdf7;
        color: #1f2933;
        box-shadow: 0 1px 2px rgba(31, 41, 51, 0.08);
    }
    .recipe-card-liked {
        border-color: #2f6b4f;
        background: #f0f8f1;
    }
    .recipe-card-disliked {
        border-color: #b42318;
        background: #fff1f0;
    }
    .recipe-card div,
    .recipe-card strong {
        color: #1f2933;
    }
    .recipe-title {
        font-size: 1.12rem;
        font-weight: 720;
        margin-bottom: 0.35rem;
        color: #1f2933;
    }
    .recipe-meta {
        color: #2f6b4f !important;
        font-size: 0.92rem;
        margin-bottom: 0.55rem;
        font-weight: 650;
    }
    .why-text {
        color: #1f2933;
        font-size: 0.95rem;
        margin-top: 0.55rem;
    }
    .small-muted {
        color: #5f6b7a !important;
        font-size: 0.86rem;
    }
    .small-muted strong {
        color: #374151 !important;
    }
    .quiz-card {
        border: 1px solid var(--meal-border);
        border-radius: 8px;
        padding: 0.75rem 0.85rem;
        margin: 0.7rem 0 0.35rem;
        background: #fffdf7;
    }
    .quiz-title {
        color: var(--meal-text);
        font-size: 0.98rem;
        font-weight: 720;
        margin-bottom: 0.25rem;
    }
    .quiz-meta {
        color: var(--meal-green);
        font-size: 0.82rem;
        font-weight: 650;
        margin-bottom: 0.35rem;
    }
    .quiz-desc {
        color: var(--meal-text);
        font-size: 0.84rem;
        line-height: 1.35;
        margin-bottom: 0.35rem;
    }
    .quiz-tags {
        color: var(--meal-muted);
        font-size: 0.8rem;
    }
    .concept-card {
        border: 1px solid var(--meal-border);
        border-radius: 8px;
        padding: 0.7rem 0.8rem;
        margin: 0.65rem 0 0.25rem;
        background: #fffdf7;
    }
    .concept-title {
        color: var(--meal-text);
        font-weight: 720;
        font-size: 0.96rem;
        margin-bottom: 0.2rem;
    }
    .concept-desc {
        color: var(--meal-muted);
        font-size: 0.84rem;
        line-height: 1.35;
    }
    .step-box {
        border-left: 4px solid var(--meal-green);
        padding: 0.35rem 0 0.35rem 0.65rem;
        margin: 0.75rem 0 0.45rem;
        background: rgba(47, 107, 79, 0.08);
        border-radius: 0 6px 6px 0;
        color: var(--meal-text);
        font-weight: 700;
    }
    .review-status {
        border: 1px solid var(--meal-border);
        border-radius: 8px;
        padding: 0.7rem 0.85rem;
        margin: 0.6rem 0 0.8rem;
        background: #fffaf0;
        color: #1f2933;
        font-weight: 650;
    }
    .review-status-ready {
        border-color: #2f6b4f;
        background: #f0f8f1;
    }
    .review-status-needed {
        border-color: #b7791f;
        background: #fff7d6;
    }
    .simple-pick-card {
        border: 1px solid #d9d2c2;
        border-radius: 10px;
        padding: 1.1rem;
        margin: 0.75rem 0;
        background: #fffdf7;
        box-shadow: 0 1px 3px rgba(31, 41, 51, 0.1);
    }
    .simple-pick-card.liked {
        border-color: #2f6b4f;
        background: #f0f8f1;
    }
    .simple-pick-card.disliked {
        border-color: #b42318;
        background: #fff1f0;
    }
    .simple-title {
        color: #1f2933 !important;
        font-size: 1.35rem;
        line-height: 1.2;
        font-weight: 800;
        margin-bottom: 0.45rem;
    }
    .simple-meta {
        color: #2f6b4f !important;
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 0.7rem;
    }
    .simple-body {
        color: #1f2933 !important;
        font-size: 0.95rem;
        line-height: 1.45;
        margin-top: 0.45rem;
    }
    .simple-body strong {
        color: #1f2933 !important;
    }
    div[data-testid="stMetric"] {
        border: 1px solid var(--meal-border);
        border-radius: 8px;
        padding: 0.75rem 0.85rem;
        background: var(--meal-panel-soft);
        box-shadow: 0 1px 2px rgba(31, 41, 51, 0.06);
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--meal-text);
    }
    div[data-testid="stSidebar"] label,
    div[data-testid="stSidebar"] p,
    div[data-testid="stSidebar"] h2,
    div[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--meal-text);
    }
    div[data-baseweb="select"] *,
    div[data-baseweb="popover"] *,
    div[data-baseweb="menu"] *,
    div[data-baseweb="radio"] *,
    div[data-baseweb="checkbox"] *,
    div[data-baseweb="slider"] * {
        color: inherit;
    }
    div[data-testid="stSlider"] label,
    div[data-testid="stSlider"] p,
    div[data-testid="stSlider"] span,
    div[data-testid="stSlider"] div,
    div[data-testid="stSlider"] [role="slider"],
    section[data-testid="stSidebar"] div[data-testid="stSlider"] *,
    section[data-testid="stSidebar"] div[data-baseweb="slider"] * {
        color: #1f2933 !important;
    }
    input,
    textarea {
        color: inherit;
    }
    div[data-testid="stCaptionContainer"] {
        color: var(--meal-muted);
    }
    div[data-testid="stAlert"] {
        border: 1px solid var(--meal-gold);
        background: #fff7d6;
        color: var(--meal-text);
    }
    .stDataFrame {
        background: var(--meal-panel);
        border: 1px solid var(--meal-border);
        border-radius: 8px;
    }
    div[data-testid="stButton"] button {
        background: #fffaf0;
        color: #1f2933;
        border: 1px solid #b8a98f;
        border-radius: 8px;
        font-weight: 700;
    }
    div[data-testid="stButton"] button:hover {
        background: #f3ead8;
        color: #1f2933;
        border-color: #2f6b4f;
    }
    div[data-testid="stButton"] button:focus,
    div[data-testid="stButton"] button:active {
        background: #efe7d7;
        color: #1f2933;
        border-color: #2f6b4f;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_artifacts() -> dict[str, pd.DataFrame]:
    recipes = pd.read_csv(DATA_DIR / "RAW_recipes.csv")
    train = pd.read_csv(DATA_DIR / "interactions_train.csv", usecols=["user_id", "recipe_id", "rating"])
    recipe_quality = pd.read_csv(OUTPUTS_DIR / "hannah_recipe_quality.csv")
    user_posteriors = pd.read_csv(OUTPUTS_DIR / "hannah_user_posteriors.csv")
    tag_posteriors = pd.read_csv(OUTPUTS_DIR / "hannah_user_tag_posteriors.csv")
    hybrid_metrics = pd.read_csv(OUTPUTS_DIR / "hannah_hybrid_metrics.csv")
    ranking_metrics = pd.read_csv(OUTPUTS_DIR / "hannah_ranking_metrics_hybrid.csv")
    calibration = pd.read_csv(OUTPUTS_DIR / "hannah_calibration.csv")
    bayes_params_path = OUTPUTS_DIR / "hannah_bayes_params.json"
    if bayes_params_path.exists():
        bayes_params = json.loads(bayes_params_path.read_text())
    else:
        bayes_params = {
            "prior_alpha": 3.8021303802071724,
            "prior_beta": 1.1978696197928274,
            "p_bar": 0.7604260760414345,
        }

    flags = build_recipe_flags(recipes).reset_index()

    recipe_table = (
        recipes[["id", "name", "minutes", "tags", "ingredients", "description", "steps", "n_steps", "n_ingredients"]]
        .merge(recipe_quality, left_on="id", right_on="recipe_id", how="inner")
        .merge(flags, on="id", how="left")
    )
    recipe_table["quality_confidence"] = 1.0 - ci_width(
        recipe_table["post_alpha"].to_numpy(),
        recipe_table["post_beta"].to_numpy(),
    )
    recipe_table["quality_confidence"] = recipe_table["quality_confidence"].clip(0, 1)
    recipe_table["tag_set"] = recipe_table["tags"].map(parse_list_column).map(set)
    recipe_table["ingredient_list"] = recipe_table["ingredients"].map(parse_list_column)
    recipe_table["step_list"] = recipe_table["steps"].map(parse_list_column)
    recipe_table["ingredient_preview"] = recipe_table["ingredient_list"].map(lambda xs: ", ".join(xs[:8]))
    recipe_table["tag_preview"] = recipe_table["tag_set"].map(format_tag_preview)
    recipe_table["description_preview"] = recipe_table["description"].map(format_description)

    return {
        "recipes": recipe_table,
        "train": train,
        "user_posteriors": user_posteriors,
        "tag_posteriors": tag_posteriors,
        "hybrid_metrics": hybrid_metrics,
        "ranking_metrics": ranking_metrics,
        "calibration": calibration,
        "bayes_params": bayes_params,
    }


@st.cache_resource(show_spinner=False)
def load_svd_model():
    model_path = MODELS_DIR / "hannah_svd.pkl"
    if not model_path.exists():
        return None
    with model_path.open("rb") as f:
        return pickle.load(f)


def beta_confidence(alpha: float, beta: float) -> float:
    return float(np.clip(1.0 - ci_width(alpha, beta), 0, 1))


def user_family_profile(user_row: pd.Series) -> dict[str, float]:
    return {
        family: float(posterior_mean(user_row[f"{family}_alpha"], user_row[f"{family}_beta"]))
        for family in FAMILIES
    }


def user_tag_profile(tag_posteriors: pd.DataFrame, user_id: int) -> dict[tuple[str, str], dict[str, float]]:
    user_tags = tag_posteriors[tag_posteriors["user_id"] == user_id]
    profile = {}
    for row in user_tags.itertuples(index=False):
        profile[(row.family, row.tag)] = {
            "mean": float(posterior_mean(row.alpha, row.beta)),
            "confidence": beta_confidence(row.alpha, row.beta),
            "n": int(row.n_exposures),
        }
    return profile


def score_recipes(
    recipes: pd.DataFrame,
    train: pd.DataFrame,
    user_row: pd.Series,
    tag_profile: dict[tuple[str, str], dict[str, float]],
    selected_dietary: list[str],
    selected_pantry: list[str],
    selected_preferences: list[tuple[str, str]] | None,
    max_minutes: int,
    top_n: int,
    include_seen: bool,
    svd_model=None,
) -> pd.DataFrame:
    user_id = int(user_row["user_id"])
    candidates = recipes[recipes["minutes"].fillna(10**9) <= max_minutes].copy()

    for label in selected_dietary:
        candidates = candidates[candidates[DIETARY_FILTERS[label]].fillna(False)]

    if selected_pantry:
        pantry_cols = [PANTRY_FILTERS[label] for label in selected_pantry]
        candidates = candidates[candidates[pantry_cols].fillna(False).any(axis=1)]

    selected_preferences = selected_preferences or []
    if selected_preferences:
        preference_filtered = filter_by_selected_preferences(candidates, selected_preferences)
        if len(preference_filtered) >= top_n:
            candidates = preference_filtered

    if not include_seen:
        seen_ids = set(train.loc[train["user_id"] == user_id, "recipe_id"])
        candidates = candidates[~candidates["id"].isin(seen_ids)]

    if candidates.empty:
        return candidates

    base_mean = float(posterior_mean(user_row["base_alpha"], user_row["base_beta"]))
    base_conf = beta_confidence(user_row["base_alpha"], user_row["base_beta"])
    family_means = user_family_profile(user_row)

    family_score = np.full(len(candidates), base_mean, dtype=float)
    family_hits = np.zeros(len(candidates), dtype=float)
    for family, mean in family_means.items():
        col = f"any_{family}"
        hit = candidates[col].fillna(False).to_numpy(dtype=bool)
        family_score[hit] += mean
        family_hits[hit] += 1
    family_score = np.where(family_hits > 0, family_score / (family_hits + 1), base_mean)

    tag_score = np.full(len(candidates), base_mean, dtype=float)
    tag_conf = np.full(len(candidates), base_conf, dtype=float)
    tag_hits = np.zeros(len(candidates), dtype=float)
    explanation_bits: list[list[tuple[str, float, int]]] = [[] for _ in range(len(candidates))]

    for family in FAMILIES:
        for tag in TAG_MEMBERS[family]:
            pref = tag_profile.get((family, tag))
            if pref is None:
                continue
            col = flag_col(family, tag)
            hit = candidates[col].fillna(False).to_numpy(dtype=bool)
            if not hit.any():
                continue
            mean = pref["mean"]
            conf = pref["confidence"]
            n_exposures = pref["n"]
            tag_score[hit] += mean
            tag_conf[hit] += conf
            tag_hits[hit] += 1
            lift = mean - base_mean
            if lift > 0:
                for idx in np.flatnonzero(hit):
                    explanation_bits[idx].append((tag, lift, n_exposures))

    tag_score = np.where(tag_hits > 0, tag_score / (tag_hits + 1), base_mean)
    tag_conf = np.where(tag_hits > 0, tag_conf / (tag_hits + 1), base_conf)

    candidates["personal_preference"] = tag_score
    candidates["family_preference"] = family_score
    bayesian_content_score = (
        BASE_WEIGHT * base_mean
        + FAMILY_WEIGHT * family_score
        + TAG_WEIGHT * tag_score
        + QUALITY_WEIGHT * candidates["quality"].to_numpy(dtype=float)
    )
    candidates["bayesian_content_score"] = bayesian_content_score
    if svd_model is not None and user_id >= 0:
        svd_predictions = np.array(
            [
                svd_model.predict(user_id, int(recipe_id)).est
                for recipe_id in candidates["id"].to_numpy()
            ],
            dtype=float,
        )
        candidates["svd_predicted_stars"] = svd_predictions
        candidates["svd_score"] = ((svd_predictions - 1.0) / 4.0).clip(0, 1)
        candidates["demo_score"] = (
            SVD_HYBRID_WEIGHT * candidates["svd_score"].to_numpy(dtype=float)
            + (1 - SVD_HYBRID_WEIGHT) * bayesian_content_score
        )
        candidates["scoring_mode"] = "SVD + Bayesian"
    else:
        candidates["svd_predicted_stars"] = np.nan
        candidates["svd_score"] = np.nan
        candidates["demo_score"] = bayesian_content_score
        candidates["scoring_mode"] = "Bayesian quick profile"
    candidates["predicted_stars"] = 1 + 4 * candidates["demo_score"].clip(0, 1)
    candidates["confidence"] = (
        0.35 * base_conf
        + 0.35 * tag_conf
        + 0.30 * candidates["quality_confidence"].to_numpy(dtype=float)
    ).clip(0, 1)
    candidates["explanation"] = [
        format_explanation(bits, base_mean) for bits in explanation_bits
    ]
    candidates["matched_preferences"] = candidates.apply(
        lambda row: matched_preference_labels(row, selected_preferences),
        axis=1,
    )

    return (
        candidates.sort_values(["demo_score", "confidence", "quality"], ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def format_explanation(bits: list[tuple[str, float, int]], base_mean: float) -> str:
    if not bits:
        return "This is a broadly strong match for the selected taste profile."
    best = sorted(bits, key=lambda x: (x[1], x[2]), reverse=True)[:3]
    parts = [f"{pretty_tag(tag)} ({n} signals)" for tag, _, n in best]
    return "Recommended because this profile tends to like " + ", ".join(parts) + "."


def filter_by_selected_preferences(
    recipes: pd.DataFrame,
    selected_preferences: list[tuple[str, str]],
) -> pd.DataFrame:
    """Keep recipes matching selected preferences.

    OR within a family, AND across families. For example, Mexican + Main dishes
    should require both a Mexican-style match and a main-dish match.
    """
    if recipes.empty or not selected_preferences:
        return recipes

    grouped: dict[str, list[str]] = {}
    for family, tag in selected_preferences:
        grouped.setdefault(family, []).append(tag)

    match = np.ones(len(recipes), dtype=bool)
    for family, tags in grouped.items():
        family_match = np.zeros(len(recipes), dtype=bool)
        for tag in tags:
            col = flag_col(family, tag)
            if col in recipes.columns:
                family_match |= recipes[col].fillna(False).to_numpy(dtype=bool)
        match &= family_match

    filtered = recipes[match].copy()
    filtered = apply_dish_exclusions(filtered, grouped)
    return filtered if not filtered.empty else recipes


def apply_dish_exclusions(recipes: pd.DataFrame, grouped_preferences: dict[str, list[str]]) -> pd.DataFrame:
    filtered = recipes
    for selected_tag in grouped_preferences.get("dish", []):
        for incompatible_tag in INCOMPATIBLE_DISH_TAGS.get(selected_tag, []):
            col = flag_col("dish", incompatible_tag)
            if col in filtered.columns:
                candidate = filtered[~filtered[col].fillna(False)].copy()
                if not candidate.empty:
                    filtered = candidate
    return filtered


def matched_preference_labels(row: pd.Series, selected_preferences: list[tuple[str, str]]) -> str:
    labels = []
    for family, tag in selected_preferences:
        col = flag_col(family, tag)
        if col in row.index and bool(row[col]):
            labels.append(pretty_tag(tag))
    return ", ".join(labels)


def pretty_tag(value: str) -> str:
    return value.replace("-", " ").replace("_", " ").title()


def pretty_recipe_name(value: str) -> str:
    return str(value).strip().title()


def format_description(value: object, max_chars: int = 150) -> str:
    if pd.isna(value):
        return "No description available yet."
    text = " ".join(str(value).strip().split())
    if not text:
        return "No description available yet."
    return text[: max_chars - 3].rstrip() + "..." if len(text) > max_chars else text


def format_tag_preview(tags: set[str], max_tags: int = 5) -> str:
    useful = [
        tag
        for tag in sorted(tags)
        if tag
        not in {
            "time-to-make",
            "course",
            "main-ingredient",
            "preparation",
            "occasion",
            "dietary",
            "easy",
        }
    ]
    if not useful:
        return "General recipe"
    return ", ".join(pretty_tag(tag) for tag in useful[:max_tags])


def confidence_label(value: float) -> str:
    if value >= 0.85:
        return "Very confident"
    if value >= 0.7:
        return "Confident"
    if value >= 0.5:
        return "Somewhat confident"
    return "Exploratory pick"


def user_label(row: pd.Series) -> str:
    n_obs = int(row["n_obs"])
    base_mean = float(posterior_mean(row["base_alpha"], row["base_beta"]))
    if n_obs >= 1000:
        history = "frequent reviewer"
    elif n_obs >= 100:
        history = "regular reviewer"
    else:
        history = "light reviewer"
    return f"Taste profile {int(row['user_id'])} - {history}, {base_mean:.0%} 5-star rate"


def render_step(label: str) -> None:
    st.markdown(f'<div class="step-box">{escape(label)}</div>', unsafe_allow_html=True)


def render_quiz_recipe_card(row: pd.Series) -> None:
    ingredients = escape(row["ingredient_preview"] or "Ingredient list unavailable")
    description = escape(row["description_preview"])
    tags = escape(row["tag_preview"])
    st.markdown(
        f"""
<div class="quiz-card">
  <div class="quiz-title">{escape(pretty_recipe_name(row["name"]))}</div>
  <div class="quiz-meta">{int(row["minutes"])} minutes | {int(row["n_ingredients"])} ingredients</div>
  <div class="quiz-desc">{description}</div>
  <div class="quiz-tags"><strong>Ingredients:</strong> {ingredients}</div>
  <div class="quiz-tags"><strong>Style:</strong> {tags}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.expander("See full recipe details"):
        st.write(row.get("description") or "No description available yet.")
        ingredients = row.get("ingredient_list") or []
        if ingredients:
            st.markdown("**Ingredients**")
            st.write(", ".join(ingredients))
        steps = row.get("step_list") or []
        if steps:
            st.markdown("**Steps**")
            for idx, step in enumerate(steps, start=1):
                st.write(f"{idx}. {step}")


def render_concept_card(concept: dict) -> None:
    st.markdown(
        f"""
<div class="concept-card">
  <div class="concept-title">Do you like {escape(concept["name"])}?</div>
  <div class="concept-desc">{escape(concept["description"])}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def collect_top_pick_feedback() -> dict[int, bool | None]:
    feedback: dict[int, bool | None] = {}
    for key, value in st.session_state.items():
        if not str(key).startswith("recipe_feedback_"):
            continue
        try:
            recipe_id = int(str(key).replace("recipe_feedback_", ""))
        except ValueError:
            continue
        feedback[recipe_id] = bool(value)
    return feedback


def collect_skipped_recipe_ids(valid_ids: set[int] | None = None) -> set[int]:
    skipped = set()
    for key, value in st.session_state.items():
        if str(key).startswith("recipe_skipped_") and value is True:
            try:
                recipe_id = int(str(key).replace("recipe_skipped_", ""))
            except ValueError:
                continue
            if valid_ids is None or recipe_id in valid_ids:
                skipped.add(recipe_id)
    return skipped


def handled_count(recs: pd.DataFrame | None = None) -> int:
    valid_ids = set(recs["id"].astype(int)) if recs is not None else None
    handled_ids = {
        int(str(key).replace("recipe_feedback_", ""))
        for key, value in st.session_state.items()
        if str(key).startswith("recipe_feedback_")
        and isinstance(value, bool)
        and (valid_ids is None or int(str(key).replace("recipe_feedback_", "")) in valid_ids)
    }
    handled_ids.update(collect_skipped_recipe_ids(valid_ids))
    return len(handled_ids)


def handled_recipe_ids(recs: pd.DataFrame | None = None) -> set[int]:
    valid_ids = set(recs["id"].astype(int)) if recs is not None else None
    ids = {
        int(str(key).replace("recipe_feedback_", ""))
        for key, value in st.session_state.items()
        if str(key).startswith("recipe_feedback_")
        and isinstance(value, bool)
        and (valid_ids is None or int(str(key).replace("recipe_feedback_", "")) in valid_ids)
    }
    ids.update(collect_skipped_recipe_ids(valid_ids))
    return ids


def next_unhandled_index(recs: pd.DataFrame, start_index: int = 0) -> int:
    handled = handled_recipe_ids(recs)
    if recs.empty:
        return 0
    for idx in range(start_index, len(recs)):
        if int(recs.iloc[idx]["id"]) not in handled:
            return idx
    for idx in range(0, min(start_index, len(recs))):
        if int(recs.iloc[idx]["id"]) not in handled:
            return idx
    return min(start_index, len(recs) - 1)


def show_recipe_detail(row: pd.Series | object) -> None:
    description = getattr(row, "description", None) if not isinstance(row, pd.Series) else row.get("description")
    st.write(description if description else "No description available yet.")

    full_ingredients = (
        getattr(row, "ingredient_list", None)
        if not isinstance(row, pd.Series)
        else row.get("ingredient_list", [])
    ) or []
    if full_ingredients:
        st.markdown("**Ingredients**")
        st.write(", ".join(full_ingredients))

    steps = (
        getattr(row, "step_list", None)
        if not isinstance(row, pd.Series)
        else row.get("step_list", [])
    ) or []
    if steps:
        st.markdown("**Steps**")
        for idx, step in enumerate(steps, start=1):
            st.write(f"{idx}. {step}")


def render_cook_screen(row: pd.Series) -> None:
    st.subheader(f"Let's Make: {pretty_recipe_name(row['name'])}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Time", f"{int(row['minutes'])} min")
    c2.metric("Ingredients", f"{int(row['n_ingredients'])}")
    if "predicted_stars" in row.index:
        c3.metric("Predicted fit", f"{float(row['predicted_stars']):.2f} stars")
    else:
        c3.metric("Recipe quality", f"{float(row.get('quality', 0)):.0%}")
    st.success("Great pick. Here is everything you need to make it.")
    show_recipe_detail(row)
    if st.button("Back to recommendations"):
        st.session_state["cook_recipe_id"] = None
        st.rerun()


def selected_recipes(recs: pd.DataFrame) -> pd.DataFrame:
    liked_ids = {
        int(key.replace("recipe_feedback_", ""))
        for key, value in st.session_state.items()
        if str(key).startswith("recipe_feedback_") and value is True
    }
    if not liked_ids:
        return recs.iloc[0:0].copy()
    return recs[recs["id"].isin(liked_ids)].sort_values(
        ["predicted_stars", "confidence"], ascending=False
    )


def render_pick_card(row: pd.Series, rank: int, total: int) -> None:
    stars = float(row["predicted_stars"])
    confidence = float(row["confidence"])
    feedback = st.session_state.get(f"recipe_feedback_{int(row['id'])}")
    skipped = bool(st.session_state.get(f"recipe_skipped_{int(row['id'])}", False))
    card_class = "simple-pick-card"
    status = ""
    if feedback is True:
        card_class += " liked"
        status = "<div class='simple-body'><strong>Status:</strong> Added to your selection</div>"
    elif feedback is False:
        card_class += " disliked"
        status = "<div class='simple-body'><strong>Status:</strong> Marked not for me</div>"
    elif skipped:
        status = "<div class='simple-body'><strong>Status:</strong> Skipped for now</div>"
    matched_preferences = row.get("matched_preferences", "") or ""
    matched_line = (
        f"<div class='simple-body'><strong>Matches:</strong> {escape(matched_preferences)}</div>"
        if matched_preferences
        else ""
    )
    scoring_mode = escape(row.get("scoring_mode", "Recommendation score"))
    svd_line = ""
    if not pd.isna(row.get("svd_predicted_stars", np.nan)):
        svd_line = (
            f"<div class='simple-body'><strong>SVD estimate:</strong> "
            f"{float(row['svd_predicted_stars']):.2f} stars</div>"
        )
    st.markdown(
        f"""
<div class="{card_class}">
  <div class="simple-title">{escape(pretty_recipe_name(row["name"]))}</div>
  <div class="simple-meta">
    Pick {rank} of {total} | {stars:.2f} predicted stars | {confidence_label(confidence)} ({confidence:.0%}) |
    {int(row["minutes"])} min | {int(row["n_ingredients"])} ingredients
  </div>
  {status}
  {matched_line}
  <div class="simple-body"><strong>Scoring:</strong> {scoring_mode}</div>
  {svd_line}
  <div class="simple-body"><strong>Why this:</strong> {escape(row["explanation"])}</div>
  <div class="simple-body"><strong>Key ingredients:</strong> {escape(row["ingredient_preview"] or "Ingredient list unavailable")}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def reviewed_count(recs: pd.DataFrame | None = None) -> int:
    valid_ids = set(recs["id"].astype(int)) if recs is not None else None
    return sum(
        1
        for key, value in st.session_state.items()
        if str(key).startswith("recipe_feedback_")
        and isinstance(value, bool)
        and (valid_ids is None or int(str(key).replace("recipe_feedback_", "")) in valid_ids)
    )


def reset_review_state_if_needed(signature: tuple) -> None:
    if st.session_state.get("review_signature") == signature:
        return
    for key in list(st.session_state.keys()):
        if (
            str(key).startswith("recipe_feedback_")
            or str(key).startswith("recipe_skipped_")
            or key in {"current_pick_index", "cook_recipe_id", "pick_warning", "review_recs"}
        ):
            st.session_state.pop(key, None)
    st.session_state["review_signature"] = signature


def preference_seed(selected_preferences: list[tuple[str, str]]) -> int:
    if not selected_preferences:
        return 17
    text = "|".join(f"{family}:{tag}" for family, tag in sorted(selected_preferences))
    return 17 + sum(ord(char) for char in text)


def choose_quick_concepts(
    selected_preferences: list[tuple[str, str]],
    selected_dietary: list[str],
    n: int = 6,
) -> list[dict]:
    selected_set = set(selected_preferences)
    selected_dish_tags = {
        tag for family, tag in selected_preferences if family == "dish"
    }
    selected_cuisine_tags = {
        tag for family, tag in selected_preferences if family == "cuisine"
    }
    blocked_ingredients = set()
    if "Vegetarian" in selected_dietary or "Vegan" in selected_dietary:
        blocked_ingredients.update({"chicken", "beef", "shrimp"})
    if "Vegan" in selected_dietary:
        blocked_ingredients.add("cheese")

    allowed = []
    for concept in QUICK_CONCEPTS:
        concept_prefs = set(concept["preferences"])
        concept_dish_tags = {tag for family, tag in concept_prefs if family == "dish"}
        if selected_dish_tags and concept_dish_tags and not (selected_dish_tags & concept_dish_tags):
            continue
        concept_cuisine_tags = {tag for family, tag in concept_prefs if family == "cuisine"}
        if selected_cuisine_tags and concept_cuisine_tags and not (selected_cuisine_tags & concept_cuisine_tags):
            continue
        ingredients = {tag for family, tag in concept_prefs if family == "ingredient"}
        if ingredients & blocked_ingredients:
            continue
        overlap = len(selected_set & concept_prefs)
        allowed.append((overlap, concept["name"], concept))

    allowed.sort(key=lambda item: (-item[0], item[1]))
    return [concept for _, _, concept in allowed[:n]]


def choose_quiz_recipes(
    recipes: pd.DataFrame,
    selected_preferences: list[tuple[str, str]] | None = None,
    selected_dietary: list[str] | None = None,
    n: int = 8,
) -> pd.DataFrame:
    selected_preferences = selected_preferences or []
    selected_dietary = selected_dietary or []
    pool = recipes[
        (recipes["minutes"].fillna(10**9) <= 90)
        & (recipes["n_ingredients"].fillna(10**9) <= 12)
        & (recipes["n"].fillna(0) >= 20)
    ].copy()
    if pool.empty:
        pool = recipes.copy()

    for label in selected_dietary:
        col = DIETARY_FILTERS[label]
        if col in pool.columns:
            pool = pool[pool[col].fillna(False)]

    if pool.empty:
        pool = recipes[
            (recipes["minutes"].fillna(10**9) <= 90)
            & (recipes["n_ingredients"].fillna(10**9) <= 12)
            & (recipes["n"].fillna(0) >= 20)
        ].copy()

    if selected_preferences:
        matched_pool = filter_by_selected_preferences(pool, selected_preferences)
        if len(matched_pool) >= max(3, n // 2):
            pool = matched_pool

    pool = pool.sort_values("quality", ascending=False).head(500)
    return pool.sample(
        n=min(n, len(pool)),
        random_state=preference_seed(selected_preferences),
    ).reset_index(drop=True)


def empty_profile_row(prior_alpha: float, prior_beta: float) -> dict[str, float]:
    row: dict[str, float] = {
        "user_id": -1,
        "n_obs": 0,
        "base_alpha": prior_alpha,
        "base_beta": prior_beta,
    }
    for family in FAMILIES:
        row[f"{family}_alpha"] = prior_alpha
        row[f"{family}_beta"] = prior_beta
    return row


def update_profile_for_recipe(
    profile_row: dict[str, float],
    tag_profile: dict[tuple[str, str], dict[str, float]],
    recipe: pd.Series,
    liked: bool,
) -> None:
    alpha_delta = 1.0 if liked else 0.0
    beta_delta = 0.0 if liked else 1.0
    profile_row["base_alpha"] += alpha_delta
    profile_row["base_beta"] += beta_delta
    profile_row["n_obs"] += 1

    for family in FAMILIES:
        family_hit = bool(recipe.get(f"any_{family}", False))
        if family_hit:
            profile_row[f"{family}_alpha"] += alpha_delta
            profile_row[f"{family}_beta"] += beta_delta

        for tag in TAG_MEMBERS[family]:
            if bool(recipe.get(flag_col(family, tag), False)):
                key = (family, tag)
                current = tag_profile.setdefault(
                    key,
                    {"alpha": profile_row[f"{family}_alpha"], "beta": profile_row[f"{family}_beta"]},
                )
                current["alpha"] += alpha_delta
                current["beta"] += beta_delta


def update_profile_for_tag(
    profile_row: dict[str, float],
    tag_profile: dict[tuple[str, str], dict[str, float]],
    family: str,
    tag: str,
    strength: float = 3.0,
) -> None:
    profile_row["base_alpha"] += 0.5
    profile_row["n_obs"] += 1
    profile_row[f"{family}_alpha"] += strength
    key = (family, tag)
    current = tag_profile.setdefault(
        key,
        {"alpha": profile_row[f"{family}_alpha"], "beta": profile_row[f"{family}_beta"]},
    )
    current["alpha"] += strength


def update_profile_for_tag_feedback(
    profile_row: dict[str, float],
    tag_profile: dict[tuple[str, str], dict[str, float]],
    family: str,
    tag: str,
    liked: bool,
    strength: float = 3.0,
) -> None:
    alpha_delta = strength if liked else 0.0
    beta_delta = 0.0 if liked else strength
    profile_row["base_alpha"] += 0.5 if liked else 0.0
    profile_row["base_beta"] += 0.0 if liked else 0.5
    profile_row["n_obs"] += 1
    profile_row[f"{family}_alpha"] += alpha_delta
    profile_row[f"{family}_beta"] += beta_delta

    key = (family, tag)
    current = tag_profile.setdefault(
        key,
        {"alpha": profile_row[f"{family}_alpha"], "beta": profile_row[f"{family}_beta"]},
    )
    current["alpha"] += alpha_delta
    current["beta"] += beta_delta


def finalize_tag_profile(raw_profile: dict[tuple[str, str], dict[str, float]]) -> dict[tuple[str, str], dict[str, float]]:
    finalized = {}
    for key, values in raw_profile.items():
        alpha = float(values["alpha"])
        beta = float(values["beta"])
        finalized[key] = {
            "mean": float(posterior_mean(alpha, beta)),
            "confidence": beta_confidence(alpha, beta),
            "n": max(1, int(round(alpha + beta))),
        }
    return finalized


def build_quick_profile(
    recipes: pd.DataFrame,
    bayes_params: dict,
    selected_preferences: list[tuple[str, str]],
    recipe_feedback: dict[int, bool | None],
    concept_feedback: dict[str, bool | None] | None = None,
) -> tuple[pd.Series, dict[tuple[str, str], dict[str, float]]]:
    prior_alpha = float(bayes_params.get("prior_alpha", 3.8021303802071724))
    prior_beta = float(bayes_params.get("prior_beta", 1.1978696197928274))
    profile_row = empty_profile_row(prior_alpha, prior_beta)
    raw_tag_profile: dict[tuple[str, str], dict[str, float]] = {}

    for family, tag in selected_preferences:
        update_profile_for_tag(profile_row, raw_tag_profile, family, tag)

    concept_feedback = concept_feedback or {}
    for concept_id, liked in concept_feedback.items():
        if liked is None:
            continue
        concept = QUICK_CONCEPT_MAP.get(concept_id)
        if concept is None:
            continue
        for family, tag in concept["preferences"]:
            update_profile_for_tag_feedback(
                profile_row,
                raw_tag_profile,
                family,
                tag,
                liked=bool(liked),
            )

    feedback_ids = [recipe_id for recipe_id, liked in recipe_feedback.items() if liked is not None]
    if feedback_ids:
        recipe_lookup = recipes.set_index("id")
        for recipe_id in feedback_ids:
            if recipe_id in recipe_lookup.index:
                update_profile_for_recipe(
                    profile_row,
                    raw_tag_profile,
                    recipe_lookup.loc[recipe_id],
                    bool(recipe_feedback[recipe_id]),
                )

    return pd.Series(profile_row), finalize_tag_profile(raw_tag_profile)


def metric_tile(label: str, value: str, caption: str) -> None:
    st.metric(label=label, value=value)
    st.caption(caption)


def render_recommendations(data: dict[str, pd.DataFrame]) -> None:
    recipes = data["recipes"]
    train = data["train"]
    user_posteriors = data["user_posteriors"]
    tag_posteriors = data["tag_posteriors"]
    svd_model = load_svd_model()

    st.markdown('<div class="app-kicker">Personalized recipe recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-title">Find recipes this customer is likely to love.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Choose a taste profile, set a few meal preferences, '
        'and MealMatch ranks recipes using past ratings, recipe features, and uncertainty-aware confidence.</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "For a new customer, use the Quick taste quiz: pick favorite food styles, react to broad food ideas, "
        "then set meal constraints. The recommendations update from those answers."
    )

    with st.sidebar:
        st.header("Build a recommendation")
        st.caption("Use this like a customer preference panel.")
        recommendation_mode = st.radio(
            "Start with",
            ["Quick taste quiz", "Saved customer profile"],
            help="Quick taste quiz is for a new customer. Saved customer profile uses an existing Food.com user.",
        )
        sorted_users = user_posteriors.sort_values("n_obs", ascending=False).reset_index(drop=True)
        user_options = sorted_users["user_id"].astype(int).tolist()
        label_map = {
            int(row.user_id): user_label(pd.Series(row._asdict()))
            for row in sorted_users.head(500).itertuples(index=False)
        }
        selected_preferences: list[tuple[str, str]] = []
        recipe_feedback: dict[int, bool | None] = {}
        concept_feedback: dict[str, bool | None] = {}
        quiz_recipes = pd.DataFrame()

        if recommendation_mode == "Saved customer profile":
            render_step("Step 1: Choose a saved customer taste profile")
            selected_user = st.selectbox(
                "Taste profile",
                user_options[:500],
                index=0,
                format_func=lambda user_id: label_map.get(int(user_id), f"Taste profile {int(user_id)}"),
                help="Each taste profile represents one real user from the Food.com training data.",
            )
        else:
            selected_user = None
            render_step("Step 1: Pick food preferences")
            st.caption("Choose anything the customer already knows they enjoy.")
            for group, choices in QUICK_TAG_CHOICES.items():
                selected_labels = st.multiselect(group, list(choices.keys()), key=f"quick_{group}")
                selected_preferences.extend(choices[label] for label in selected_labels)

            render_step("Step 2: Set must-have dietary needs")
            selected_dietary = st.multiselect("Dietary needs", list(DIETARY_FILTERS.keys()))

            render_step("Step 3: React to broad food ideas")
            st.caption(
                "These quick answers teach the model general taste patterns. Love it adds positive evidence, "
                "Not for me adds negative evidence, and Skip adds no evidence."
            )
            concepts_to_show = choose_quick_concepts(selected_preferences, selected_dietary)
            for concept in concepts_to_show:
                render_concept_card(concept)
                choice = st.radio(
                    "Your reaction",
                    list(RATING_OPTIONS.keys()),
                    horizontal=True,
                    key=f"concept_{concept['id']}",
                    label_visibility="visible",
                )
                concept_feedback[concept["id"]] = RATING_OPTIONS[choice]

        render_step("Step 4: Set meal constraints")
        max_minutes = st.slider("Maximum cooking time", 15, 240, 90, step=15)
        if recommendation_mode == "Saved customer profile":
            selected_dietary = st.multiselect("Dietary needs", list(DIETARY_FILTERS.keys()))
        selected_pantry = st.multiselect(
            "Ingredients to require",
            list(PANTRY_FILTERS.keys()),
            help="If you choose ingredients, recommendations must include at least one of them.",
        )
        top_n = st.slider("Number of recipes", 5, 25, 10)
        if recommendation_mode == "Saved customer profile":
            include_seen = st.toggle(
                "Show recipes already rated by this profile",
                value=False,
                help="Usually off for customer recommendations.",
            )
        else:
            include_seen = False

    review_signature = (
        recommendation_mode,
        int(selected_user) if selected_user is not None else None,
        tuple(sorted(selected_preferences)),
        tuple(sorted(selected_dietary)),
        tuple(sorted(selected_pantry)),
        int(max_minutes),
        int(top_n),
        bool(include_seen),
        tuple(sorted((key, str(value)) for key, value in concept_feedback.items())),
    )
    reset_review_state_if_needed(review_signature)
    if recommendation_mode == "Quick taste quiz":
        recipe_feedback = collect_top_pick_feedback()
    else:
        recipe_feedback = {}

    if recommendation_mode == "Saved customer profile":
        user_row = user_posteriors[user_posteriors["user_id"] == selected_user].iloc[0]
        tag_profile = user_tag_profile(tag_posteriors, int(selected_user))
        scoring_train = train
        profile_caption = "Past ratings used for this profile"
    else:
        user_row, tag_profile = build_quick_profile(
            recipes=recipes,
            bayes_params=data["bayes_params"],
            selected_preferences=selected_preferences,
            recipe_feedback=recipe_feedback,
            concept_feedback=concept_feedback,
        )
        scoring_train = train
        profile_caption = "Quick likes, dislikes, and preference choices"

    recs = score_recipes(
        recipes=recipes,
        train=scoring_train,
        user_row=user_row,
        tag_profile=tag_profile,
        selected_dietary=selected_dietary,
        selected_pantry=selected_pantry,
        selected_preferences=selected_preferences if recommendation_mode == "Quick taste quiz" else None,
        max_minutes=max_minutes,
        top_n=top_n,
        include_seen=include_seen,
        svd_model=svd_model if recommendation_mode == "Saved customer profile" else None,
    )
    if "review_recs" not in st.session_state:
        st.session_state["review_recs"] = recs.copy()
    else:
        recs = st.session_state["review_recs"].copy()

    base_mean = float(posterior_mean(user_row["base_alpha"], user_row["base_beta"]))
    st.subheader("Recommendation Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_tile("Taste signals", f"{int(user_row['n_obs']):,}", profile_caption)
    with c2:
        metric_tile("5-star tendency", f"{base_mean:.0%}", "How often this profile gives 5 stars")
    with c3:
        metric_tile("Recipes searched", f"{len(recipes):,}", "Recipes available to rank")
    with c4:
        metric_tile("Matches found", f"{len(recs)}", "After your filters")

    if recs.empty:
        st.warning("No recipes matched those filters. Try increasing cooking time or removing one preference.")
        return

    st.subheader("Top Picks")
    st.caption("Choose Like, Skip, or Not for me. The app moves to the next recipe automatically.")

    cook_recipe_id = st.session_state.get("cook_recipe_id")
    if cook_recipe_id is not None:
        recipe_match = recipes[recipes["id"] == int(cook_recipe_id)]
        if not recipe_match.empty:
            render_cook_screen(recipe_match.iloc[0])
            return
        st.session_state["cook_recipe_id"] = None

    review_total = min(top_n, len(recs))
    review_complete = handled_count(recs) >= review_total
    if review_complete:
        st.success("Review complete. Choose a recipe from your selection to cook.")
    else:
        current_index = int(st.session_state.get("current_pick_index", 0))
        if current_index >= len(recs):
            current_index = 0
            st.session_state["current_pick_index"] = 0
        current_index = next_unhandled_index(recs, current_index)
        st.session_state["current_pick_index"] = current_index

        current_row = recs.iloc[current_index]
        display_rank = min(handled_count(recs) + 1, review_total)
        render_pick_card(current_row, display_rank, review_total)

        with st.expander("See full recipe details"):
            show_recipe_detail(current_row)

        like_col, skip_col, dislike_col = st.columns(3)
        with like_col:
            if st.button("Like this", use_container_width=True):
                st.session_state.pop("pick_warning", None)
                st.session_state.pop(f"recipe_skipped_{int(current_row['id'])}", None)
                st.session_state[f"recipe_feedback_{int(current_row['id'])}"] = True
                st.session_state["current_pick_index"] = next_unhandled_index(recs, current_index + 1)
                st.rerun()
        with skip_col:
            if st.button("Skip for now", use_container_width=True):
                st.session_state.pop("pick_warning", None)
                st.session_state.pop(f"recipe_feedback_{int(current_row['id'])}", None)
                st.session_state[f"recipe_skipped_{int(current_row['id'])}"] = True
                st.session_state["current_pick_index"] = next_unhandled_index(recs, current_index + 1)
                st.rerun()
        with dislike_col:
            if st.button("Not for me", use_container_width=True):
                st.session_state.pop("pick_warning", None)
                st.session_state.pop(f"recipe_skipped_{int(current_row['id'])}", None)
                st.session_state[f"recipe_feedback_{int(current_row['id'])}"] = False
                st.session_state["current_pick_index"] = next_unhandled_index(recs, current_index + 1)
                st.rerun()

    chosen = selected_recipes(recs)
    skipped_count = len(collect_skipped_recipe_ids(set(recs["id"].astype(int))))
    st.subheader("Your Selection")
    if chosen.empty:
        st.caption(f"{reviewed_count(recs)} recipe(s) rated, {skipped_count} skipped. Like a recipe to add it here.")
    else:
        st.caption(f"{len(chosen)} selected, {reviewed_count(recs)} rated, {skipped_count} skipped.")
        best = chosen.iloc[0]
        st.markdown(
            f"**App recommends:** {pretty_recipe_name(best['name'])} "
            f"({float(best['predicted_stars']):.2f} predicted stars, {float(best['confidence']):.0%} confidence)"
        )
        st.caption("Click any recipe row below if you want to cook a different selected recipe.")
        selection_table = chosen[["name", "predicted_stars", "confidence", "minutes"]].copy()
        selection_table["name"] = selection_table["name"].map(pretty_recipe_name)
        selection_event = st.dataframe(
            selection_table.rename(
                columns={
                    "name": "Recipe",
                    "predicted_stars": "Predicted stars",
                    "confidence": "Confidence",
                    "minutes": "Minutes",
                }
            ),
            use_container_width=True,
            hide_index=True,
            key="selected_recipe_table",
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "Predicted stars": st.column_config.NumberColumn(format="%.2f"),
                "Confidence": st.column_config.ProgressColumn(format="%.0f", min_value=0, max_value=1),
            },
        )
        selected_rows = selection_event.selection.rows if selection_event else []
        recipe_to_cook = chosen.iloc[selected_rows[0]] if selected_rows else best
        st.caption(f"Ready to cook: {pretty_recipe_name(recipe_to_cook['name'])}")
        if st.button("Cook selected recipe", use_container_width=True):
            st.session_state[f"recipe_feedback_{int(recipe_to_cook['id'])}"] = True
            st.session_state["cook_recipe_id"] = int(recipe_to_cook["id"])
            st.rerun()

    with st.expander("Compare recommendations in a table"):
        display_cols = [
            "name",
            "predicted_stars",
            "svd_predicted_stars",
            "confidence",
            "minutes",
            "n",
            "scoring_mode",
            "explanation",
            "ingredient_preview",
        ]
        table = recs[display_cols].rename(
            columns={
                "name": "Recipe",
                "predicted_stars": "Predicted stars",
                "svd_predicted_stars": "SVD stars",
                "confidence": "Confidence",
                "minutes": "Minutes",
                "n": "Recipe ratings",
                "scoring_mode": "Scoring",
                "explanation": "Why",
                "ingredient_preview": "Ingredients",
            }
        )
        table["Recipe"] = table["Recipe"].map(pretty_recipe_name)
        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Predicted stars": st.column_config.NumberColumn(format="%.2f"),
                "SVD stars": st.column_config.NumberColumn(format="%.2f"),
                "Confidence": st.column_config.ProgressColumn(format="%.0f", min_value=0, max_value=1),
            },
        )


def render_metrics(data: dict[str, pd.DataFrame]) -> None:
    st.header("Model Report: Fixed Notebook Results")
    st.write(
        "This tab is not a live recommendation screen. It is the fixed report card from the notebooks, "
        "so these numbers stay the same while you click around in Find Recipes."
    )
    metrics = data["hybrid_metrics"]
    ranking = data["ranking_metrics"]
    calibration = data["calibration"]

    stars = metrics[(metrics["track"] == "stars") & (metrics["metric"].isin(["RMSE", "MAE"]))]
    cold_test = stars[stars["split"] == "test"]
    warm = stars[stars["split"] == "warm_holdout"]

    st.subheader("What The App Uses")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
**Saved customer profile**

This is for a user who already exists in the Food.com data. The app knows their past ratings, so it can use:

- **SVD**: what similar users liked
- **Bayesian preferences**: what tags/ingredients this user tends to like
"""
        )
    with c2:
        st.markdown(
            """
**Quick taste quiz**

This is for a brand-new user. The app does not know their rating history yet, so it uses:

- quiz answers
- likes / dislikes / skips
- Bayesian updating
- historical recipe quality
"""
        )

    st.subheader("How Recommendations Are Calculated")
    st.markdown(
        f"""
The app does **not** choose a different model every time. It uses fixed scoring rules.

**Saved Customer Profile**

1. Load the trained SVD model from `outputs/models/hannah_svd.pkl`.
2. Predict how the saved user would rate each candidate recipe.
3. Compute a Bayesian/content score from the user's learned tag preferences and the recipe's historical quality.
4. Blend them using the notebook's chosen hybrid weight.

```text
final score = {SVD_HYBRID_WEIGHT:.0%} * SVD score + {1 - SVD_HYBRID_WEIGHT:.0%} * Bayesian/content score
predicted stars = 1 + 4 * final score
```

**Why {SVD_HYBRID_WEIGHT:.0%} / {1 - SVD_HYBRID_WEIGHT:.0%}?**

Hannah's hybrid notebook tested different SVD/Bayesian blend weights on validation data.
The best star-rating blend was `w = 0.70`, meaning 70% SVD and 30% Bayesian/content.
So the app uses that same notebook-backed weight instead of choosing a new one randomly.

**Quick Taste Quiz**

1. Start with the Bayesian prior from the notebooks.
2. Add positive evidence for `Love it`.
3. Add negative evidence for `Not for me`.
4. Add no evidence for `Skip`.
5. Score recipes using the temporary Bayesian profile plus historical recipe quality:

```text
Bayesian/content score =
  baseline user tendency
  + tag and ingredient preference matches
  + recipe quality from historical ratings
```

Quick quiz does not use SVD because the new user has no historical SVD user factor yet.
"""
    )

    st.subheader("Rating Prediction")
    st.info(
        "These results do not change when you click around in Find Recipes. "
        "They are fixed test sets from the notebooks, used only to prove how well the models worked."
    )
    st.markdown(
        """
**Simple version:** we test the recommender in two situations.

"""
    )
    split_explainer = pd.DataFrame(
        [
            {
                "Test set": "Warm holdout",
                "Plain meaning": "The easier case: the model has seen history for this kind of user/recipe before.",
                "Why we keep it": "Shows how well the model works for returning customers and familiar recipes.",
            },
            {
                "Test set": "Cold test",
                "Plain meaning": "The harder case: the recipe is newer or has less history.",
                "Why we keep it": "Shows whether Bayesian recipe quality and content still help when SVD has less information.",
            },
        ]
    )
    st.dataframe(split_explainer, use_container_width=True, hide_index=True)
    st.markdown(
        """
**Why have both?** Because a customer app has both problems:

- Some users/recipes have lots of history, where SVD is strong.
- Some recipes have little history, where Bayesian priors/content are more useful.

For the presentation, you can say: **warm = familiar case, cold = harder new-recipe case.**

**RMSE/MAE** are prediction error scores. Lower is better.
"""
    )

    with st.expander("Show the detailed notebook score tables"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Cold test set**")
            cold_table = cold_test.pivot_table(index="model", columns="metric", values="value").round(4)
            st.dataframe(cold_table, use_container_width=True)
        with c2:
            st.markdown("**Warm holdout set**")
            warm_table = warm.pivot_table(index="model", columns="metric", values="value").round(4)
            st.dataframe(warm_table, use_container_width=True)

    if {"model", "metric", "value"}.issubset(cold_test.columns):
        cold_rmse = cold_test[cold_test["metric"] == "RMSE"].set_index("model")["value"]
        if "SVD" in cold_rmse.index and "hybrid w* (1+4p)" in cold_rmse.index:
            st.info(
                f"Notebook headline: on cold test RMSE, hybrid improves over SVD "
                f"({cold_rmse['hybrid w* (1+4p)']:.4f} vs {cold_rmse['SVD']:.4f})."
            )

    st.subheader("Top-10 Ranking")
    st.markdown(
        """
These metrics ask: **when there is one recipe we know the user loved, did the model put it near the top?**

- **HR@10**: whether the loved recipe appeared in the top 10.
- **NDCG@10**: gives more credit when the loved recipe appears closer to #1.
"""
    )
    rank_view = ranking[["model", "hr_at_10", "ndcg_at_10", "n_users", "protocol"]].copy()
    rank_view["hr_at_10"] = rank_view["hr_at_10"].round(3)
    rank_view["ndcg_at_10"] = rank_view["ndcg_at_10"].round(3)
    focus_models = [
        "svd",
        "bayes",
        "hybrid_wstar (w=0.70)",
        "popularity",
        "pop_blend (lambda=2)",
        "bayes_cold_content",
        "random_cold",
    ]
    rank_view = rank_view[rank_view["model"].isin(focus_models)]
    st.dataframe(rank_view, use_container_width=True, hide_index=True)

    st.subheader("Confidence Calibration")
    st.caption("Calibration checks whether confidence numbers are believable, not just large.")
    ece = calibration[calibration["row_type"] == "summary"].copy()
    if ece.empty:
        ece = calibration.groupby("split", as_index=False)[["stated", "realized"]].mean()
    calibration_view = ece.copy()
    for col in ["stated", "realized", "ece_raw", "ece_recal"]:
        if col in calibration_view.columns:
            calibration_view[col] = calibration_view[col].round(4)
    st.dataframe(calibration_view, use_container_width=True, hide_index=True)

    st.subheader("Honest Takeaways")
    st.markdown(
        """
- SVD is useful for saved users because it has historical user-recipe factors.
- The Bayesian layer makes quick feedback and explanations possible.
- The hybrid improves cold rating prediction in the notebook results, so the app uses the notebook's 70/30 SVD/Bayesian blend for saved profiles.
- Popularity remains very strong for pure top-N ranking, so the app emphasizes personalization and explanation rather than claiming to beat popularity everywhere.
"""
    )


def render_handoff() -> None:
    st.header("About This Demo")
    st.markdown(
        """
- This app is a customer-facing demo for the Bayesian recipe recommendation project.
- It uses saved model outputs rather than retraining models while the customer waits.
- In quick-quiz mode, likes/dislikes immediately update a temporary Bayesian taste profile.
- Saved customer profiles use the regenerated Surprise SVD model plus Bayesian tag preferences.
- Quick-quiz users do not have trained SVD factors yet, so they use Bayesian feedback plus historical recipe quality.
- The confidence score is lower when the model has less evidence.
- The model-results tab is included for the class report and team review.
"""
    )


def main() -> None:
    data = load_artifacts()
    tabs = st.tabs(["Find Recipes", "Model Report", "About"])
    with tabs[0]:
        render_recommendations(data)
    with tabs[1]:
        render_metrics(data)
    with tabs[2]:
        render_handoff()


if __name__ == "__main__":
    main()
