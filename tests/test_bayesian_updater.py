import numpy as np
import pytest

from src.models.bayesian_updater import (
    BetaPreference,
    ci_width,
    posterior_mean,
    posterior_var,
)


def test_beta_preference_update_increases_mean_after_like():
    preference = BetaPreference()
    before = preference.mean
    preference.update(liked=True)
    assert preference.mean > before


def test_conjugate_update_identity():
    # after k likes out of n observations, posterior is Beta(a0+k, b0+n-k)
    a0, b0 = 2.0, 3.0
    outcomes = [True, False, True, True, False, True]
    pref = BetaPreference(alpha=a0, beta=b0)
    for liked in outcomes:
        pref.update(liked=liked)
    k = sum(outcomes)
    n = len(outcomes)
    assert pref.alpha == a0 + k
    assert pref.beta == b0 + (n - k)
    assert pref.mean == pytest.approx(posterior_mean(a0 + k, b0 + n - k))
    assert pref.variance == pytest.approx(posterior_var(a0 + k, b0 + n - k))


def test_posterior_mean_var_known_values():
    assert posterior_mean(2, 2) == pytest.approx(0.5)
    assert posterior_var(2, 2) == pytest.approx(4 / (16 * 5))
    assert posterior_mean(1, 3) == pytest.approx(0.25)


def test_batch_safe_helpers():
    a = np.array([1.0, 2.0, 10.0])
    b = np.array([1.0, 2.0, 30.0])
    means = posterior_mean(a, b)
    assert isinstance(means, np.ndarray)
    np.testing.assert_allclose(means, [0.5, 0.5, 0.25])
    widths = ci_width(a, b)
    assert isinstance(widths, np.ndarray)
    for i in range(3):
        assert widths[i] == pytest.approx(ci_width(a[i], b[i]))
    # scalar input returns plain floats
    assert isinstance(posterior_mean(1, 1), float)
    assert isinstance(posterior_var(1, 1), float)
    assert isinstance(ci_width(1, 1), float)


def test_confidence_low_for_prior_only():
    # Beta(1,1): central 95% CI is (0.025, 0.975), width 0.95 -> confidence 0.05.
    # The old 1 - sqrt(variance) definition reported ~0.83 here.
    pref = BetaPreference()
    assert pref.confidence == pytest.approx(0.05, abs=1e-9)
    assert pref.confidence < 0.1


def test_confidence_monotone_in_evidence():
    # same posterior mean, growing evidence -> strictly increasing confidence
    confidences = [BetaPreference(alpha=s, beta=s).confidence
                   for s in [1, 3, 11, 51, 201]]
    assert all(b > a for a, b in zip(confidences, confidences[1:]))


def test_confidence_grows_with_updates():
    pref = BetaPreference()
    prev = pref.confidence
    rng = np.random.default_rng(42)
    for _ in range(30):
        pref.update(liked=bool(rng.integers(0, 2)))
        assert pref.confidence >= prev - 1e-12
        prev = pref.confidence


def test_confidence_bounded():
    for a, b in [(1, 1), (0.5, 0.5), (1, 20), (20, 1), (100, 100), (5000, 1)]:
        c = BetaPreference(alpha=a, beta=b).confidence
        assert 0.0 <= c <= 1.0


def test_ci_width_matches_scipy_interval():
    from scipy import stats

    lo, hi = stats.beta.interval(0.95, 4.0, 6.0)
    assert ci_width(4.0, 6.0) == pytest.approx(hi - lo)
    lo90, hi90 = stats.beta.interval(0.90, 4.0, 6.0)
    assert ci_width(4.0, 6.0, level=0.90) == pytest.approx(hi90 - lo90)
