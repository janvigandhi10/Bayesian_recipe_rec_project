from dataclasses import dataclass

import pandas as pd


@dataclass
class SVDConfig:
    n_factors: int = 100
    n_epochs: int = 20
    lr_all: float = 0.005
    reg_all: float = 0.02
    random_state: int = 42


def prepare_surprise_dataset(interactions: pd.DataFrame):
    """Create a Surprise dataset from user, recipe, and rating columns."""
    from surprise import Dataset, Reader

    reader = Reader(rating_scale=(1, 5))
    return Dataset.load_from_df(interactions[["user_id", "recipe_id", "rating"]], reader)


def train_svd(interactions: pd.DataFrame, config: SVDConfig | None = None):
    """Train an SVD model with the Surprise library."""
    from surprise import SVD

    config = config or SVDConfig()
    data = prepare_surprise_dataset(interactions)
    trainset = data.build_full_trainset()
    model = SVD(
        n_factors=config.n_factors,
        n_epochs=config.n_epochs,
        lr_all=config.lr_all,
        reg_all=config.reg_all,
        random_state=config.random_state,
    )
    model.fit(trainset)
    return model

