import pandas as pd


def normalize_rating(predicted_rating: float, min_rating: float = 1, max_rating: float = 5) -> float:
    """Map a predicted rating onto a 0-1 scale."""
    return (predicted_rating - min_rating) / (max_rating - min_rating)


def hybrid_score(svd_score: float, bayesian_score: float, svd_weight: float = 0.7) -> float:
    """Combine normalized SVD score and Bayesian preference score."""
    bayesian_weight = 1 - svd_weight
    return svd_weight * svd_score + bayesian_weight * bayesian_score


def attach_hybrid_scores(
    recommendations: pd.DataFrame,
    svd_col: str = "svd_score",
    bayesian_col: str = "bayesian_score",
    svd_weight: float = 0.7,
) -> pd.DataFrame:
    """Add a hybrid score column to a recommendation dataframe."""
    scored = recommendations.copy()
    scored["hybrid_score"] = scored.apply(
        lambda row: hybrid_score(row[svd_col], row[bayesian_col], svd_weight=svd_weight),
        axis=1,
    )
    return scored.sort_values("hybrid_score", ascending=False)

