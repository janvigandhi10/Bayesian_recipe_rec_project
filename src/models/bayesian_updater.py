"""Beta-Binomial preference state with evidence-honest confidence.

Confidence is defined as ``1 - width of the central 95% credible interval``
(``scipy.stats.beta.interval``). This is evidence-honest: a prior-only
posterior such as Beta(1, 1) has a wide interval (width 0.95), so confidence
is low (~0.05), and confidence grows only as observed evidence tightens the
posterior. The previous definition (``1 - posterior sd``) reported ~0.83 for
a user with zero observations, overstating certainty.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats


def posterior_mean(alpha, beta):
    """Mean of Beta(alpha, beta). Batch-safe: scalars or numpy arrays."""
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    out = alpha / (alpha + beta)
    return float(out) if out.ndim == 0 else out


def posterior_var(alpha, beta):
    """Variance of Beta(alpha, beta). Batch-safe: scalars or numpy arrays."""
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    total = alpha + beta
    out = (alpha * beta) / ((total**2) * (total + 1))
    return float(out) if out.ndim == 0 else out


def ci_width(alpha, beta, level=0.95):
    """Width of the central credible interval of Beta(alpha, beta).

    Batch-safe: scalars or numpy arrays (returns an array for array input).
    """
    lo, hi = stats.beta.interval(level, np.asarray(alpha, dtype=float),
                                 np.asarray(beta, dtype=float))
    out = np.asarray(hi) - np.asarray(lo)
    return float(out) if out.ndim == 0 else out


@dataclass
class BetaPreference:
    alpha: float = 1.0
    beta: float = 1.0

    @property
    def mean(self) -> float:
        return posterior_mean(self.alpha, self.beta)

    @property
    def variance(self) -> float:
        return posterior_var(self.alpha, self.beta)

    @property
    def confidence(self) -> float:
        """1 - (95% credible-interval width), in [0, 1].

        Evidence-honest: a prior-only posterior has a wide CI and therefore a
        LOW confidence; more observations shrink the CI and raise it.
        """
        return float(1.0 - ci_width(self.alpha, self.beta, level=0.95))

    def update(self, liked: bool) -> "BetaPreference":
        if liked:
            self.alpha += 1
        else:
            self.beta += 1
        return self
