from src.evaluation.metrics import mae, rmse


def test_rmse():
    assert rmse([1, 2, 3], [1, 2, 4]) == 0.5773502691896257


def test_mae():
    assert mae([1, 2, 3], [1, 2, 4]) == 0.3333333333333333

