from pathlib import Path

import pandas as pd


def load_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """Load a CSV file from the project data directory or an explicit path."""
    return pd.read_csv(path, **kwargs)


def load_interaction_splits(data_dir: str | Path = "Data") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test interaction splits."""
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "interactions_train.csv")
    validation = pd.read_csv(data_dir / "interactions_validation.csv")
    test = pd.read_csv(data_dir / "interactions_test.csv")
    return train, validation, test

