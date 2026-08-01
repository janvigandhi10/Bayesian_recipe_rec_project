from src.models.bayesian_updater import BetaPreference


def test_beta_preference_update_increases_mean_after_like():
    preference = BetaPreference()
    before = preference.mean
    preference.update(liked=True)
    assert preference.mean > before

