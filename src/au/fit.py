"""Maximum-likelihood fitting with several random starts."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .models import Model, neg_log_lik
from .task import Task


@dataclass
class Fit:
    theta: np.ndarray
    nll: float
    k: int
    n: int

    @property
    def aic(self) -> float:
        return 2 * self.nll + 2 * self.k

    @property
    def bic(self) -> float:
        return 2 * self.nll + self.k * np.log(self.n)


def fit(model: Model, task: Task, choices, outcomes, rng, n_starts: int = 5) -> Fit:
    """Minimise the negative log-likelihood from n_starts random points; keep the best."""
    best = None
    lo = np.array([b[0] for b in model.bounds])
    hi = np.array([b[1] for b in model.bounds])
    for _ in range(n_starts):
        x0 = lo + rng.random(len(lo)) * (hi - lo)
        res = minimize(neg_log_lik, x0, args=(task, choices, outcomes),
                       method="L-BFGS-B", bounds=model.bounds)
        if best is None or res.fun < best.fun:
            best = res
    return Fit(best.x, float(best.fun), len(model.params), task.n_trials)
