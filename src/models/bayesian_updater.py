from dataclasses import dataclass

import numpy as np


@dataclass
class BetaPreference:
    alpha: float = 1.0
    beta: float = 1.0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / ((total**2) * (total + 1))

    @property
    def confidence(self) -> float:
        return float(1 - np.sqrt(self.variance))

    def update(self, liked: bool) -> "BetaPreference":
        if liked:
            self.alpha += 1
        else:
            self.beta += 1
        return self

