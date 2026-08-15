import numpy as np
import pytest

from src.evaluation.hannah_metrics import (
    hr_at_k,
    mae,
    mean_ci_normal,
    ndcg_at_k,
    rank_of_positive,
    rmse,
    wilson_ci,
)


def test_rmse_mae_reexported():
    assert rmse([1, 2, 3], [1, 2, 4]) == pytest.approx(1 / np.sqrt(3))
    assert mae([1, 2, 3], [1, 2, 4]) == pytest.approx(1 / 3)


def test_wilson_known_value():
    # 5/10 at 95%: standard Wilson interval (0.2366, 0.7634)
    lo, hi = wilson_ci(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3)
    assert hi == pytest.approx(0.7634, abs=1e-3)


def test_wilson_edge_cases():
    assert wilson_ci(0, 0) == (0.0, 1.0)
    lo0, hi0 = wilson_ci(0, 10)
    assert lo0 == 0.0
    assert 0.0 < hi0 < 0.35
    lo1, hi1 = wilson_ci(10, 10)
    assert hi1 == pytest.approx(1.0)
    assert 0.65 < lo1 < 1.0
    # more evidence at the same rate -> narrower interval
    lo_small, hi_small = wilson_ci(5, 10)
    lo_big, hi_big = wilson_ci(500, 1000)
    assert (hi_big - lo_big) < (hi_small - lo_small)
    # interval always within [0, 1]
    for k, n in [(0, 1), (1, 1), (3, 7), (99, 100)]:
        lo, hi = wilson_ci(k, n)
        assert 0.0 <= lo <= hi <= 1.0


def test_mean_ci_normal_known_value():
    x = [1, 2, 3, 4, 5]
    lo, hi = mean_ci_normal(x)
    # mean 3, sd sqrt(2.5), se sqrt(0.5), z 1.95996 -> half-width 1.38590
    assert lo == pytest.approx(3 - 1.38590, abs=1e-4)
    assert hi == pytest.approx(3 + 1.38590, abs=1e-4)


def test_mean_ci_normal_degenerate():
    lo, hi = mean_ci_normal([])
    assert np.isnan(lo) and np.isnan(hi)
    assert mean_ci_normal([2.5]) == (2.5, 2.5)


def test_rank_of_positive_no_ties():
    assert rank_of_positive(0.9, [0.1, 0.5]) == 1
    assert rank_of_positive(0.3, [0.1, 0.5]) == 2
    assert rank_of_positive(0.0, [0.1, 0.5]) == 3


def test_rank_of_positive_ties_conservative_without_rng():
    # both tied negatives rank ahead of the positive
    assert rank_of_positive(0.5, [0.5, 0.5, 0.1]) == 3


def test_rank_of_positive_ties_random_with_rng():
    ranks = {rank_of_positive(0.5, [0.5, 0.5, 0.1],
                              rng=np.random.default_rng(seed))
             for seed in range(200)}
    # positive lands uniformly among the ties: every slot 1..3 reachable
    assert ranks == {1, 2, 3}


def test_hr_at_k():
    assert hr_at_k(1) == 1.0
    assert hr_at_k(10) == 1.0
    assert hr_at_k(11) == 0.0
    assert hr_at_k(3, k=2) == 0.0
    np.testing.assert_array_equal(hr_at_k(np.array([1, 10, 11])),
                                  np.array([1.0, 1.0, 0.0]))


def test_ndcg_at_k_known_values():
    assert ndcg_at_k(1) == pytest.approx(1.0)
    assert ndcg_at_k(2) == pytest.approx(1 / np.log2(3))
    assert ndcg_at_k(10) == pytest.approx(1 / np.log2(11))
    assert ndcg_at_k(11) == 0.0
    got = ndcg_at_k(np.array([1, 2, 11]))
    np.testing.assert_allclose(got, [1.0, 1 / np.log2(3), 0.0])


def test_hr_ndcg_consistency():
    # ndcg positive exactly where hr positive
    ranks = np.arange(1, 30)
    assert ((ndcg_at_k(ranks, k=10) > 0) == (hr_at_k(ranks, k=10) > 0)).all()
