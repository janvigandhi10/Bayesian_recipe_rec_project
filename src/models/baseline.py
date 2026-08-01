import pandas as pd


def global_mean_prediction(train: pd.DataFrame) -> float:
    """Return the global mean rating as the simplest baseline."""
    return float(train["rating"].mean())

