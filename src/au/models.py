"""Two models of how a participant chooses SAFE vs BET.

Both models learn each machine's win probability from experience:

    p_hat[m] <- p_hat[m] + alpha * (outcome - p_hat[m])      (only after a BET)

and choose with a logistic (softmax) rule:

    P(BET) = sigmoid(beta * (Q_bet - Q_safe))

EV-RL (no anticipation)
    Q_bet  = p_hat[m] * hi + bias
    Q_safe = safe

AU-RL (anticipatory utility)
    Q_bet  = p_hat[m] * hi * (1 + kappa * f(delay)) + bias
    Q_safe = safe

    f(delay) = delay / 16 s, so kappa is the extra value, as a fraction of the
    prize, that a win is worth when you get to savour it for the longest delay.
    kappa > 0 means savouring (people like waiting for a win), kappa < 0 dread.

The bias term is a general taste for gambling. It shifts BET up on every
machine equally, so it cannot explain a preference that changes with delay.
Only kappa can do that, which is what makes the two ideas separable.
"""
from dataclasses import dataclass

import numpy as np

from .task import P_WIN, Task

D_MAX = 16.0
P0 = 0.5   # prior belief about each machine's win probability


@dataclass(frozen=True)
class Model:
    name: str
    params: tuple       # parameter names, in order
    bounds: tuple       # (low, high) per parameter


EV = Model("EV-RL", ("alpha", "beta", "bias"), ((0.0, 1.0), (0.0, 20.0), (-2.0, 2.0)))
AU = Model("AU-RL", ("alpha", "beta", "bias", "kappa"),
           ((0.0, 1.0), (0.0, 20.0), (-2.0, 2.0), (-1.0, 2.0)))
MODELS = {"EV": EV, "AU": AU}


def _unpack(theta):
    alpha, beta, bias = theta[0], theta[1], theta[2]
    kappa = theta[3] if len(theta) > 3 else 0.0
    return alpha, beta, bias, kappa


def p_bet(theta, task: Task, choices, outcomes):
    """Model's P(BET) on every trial, given what the participant actually saw."""
    alpha, beta, bias, kappa = _unpack(theta)
    p_hat = np.full(3, P0)
    out = np.empty(task.n_trials)
    for t in range(task.n_trials):
        m = task.machine[t]
        q_bet = p_hat[m] * task.hi[t] * (1 + kappa * task.delay[t] / D_MAX) + bias
        out[t] = 1.0 / (1.0 + np.exp(-beta * (q_bet - task.safe[t])))
        if choices[t] == 1:                       # outcome only seen after a BET
            p_hat[m] += alpha * (outcomes[t] - p_hat[m])
    return out


def neg_log_lik(theta, task: Task, choices, outcomes):
    p = np.clip(p_bet(theta, task, choices, outcomes), 1e-9, 1 - 1e-9)
    return -np.sum(choices * np.log(p) + (1 - choices) * np.log(1 - p))


def simulate(theta, task: Task, rng: np.random.Generator):
    """Play the task once. Returns choices (1 = BET) and outcomes (1 = win, nan if SAFE)."""
    alpha, beta, bias, kappa = _unpack(theta)
    p_hat = np.full(3, P0)
    choices = np.zeros(task.n_trials, int)
    outcomes = np.full(task.n_trials, np.nan)
    for t in range(task.n_trials):
        m = task.machine[t]
        q_bet = p_hat[m] * task.hi[t] * (1 + kappa * task.delay[t] / D_MAX) + bias
        pb = 1.0 / (1.0 + np.exp(-beta * (q_bet - task.safe[t])))
        choices[t] = rng.random() < pb
        if choices[t]:
            outcomes[t] = float(rng.random() < P_WIN)
            p_hat[m] += alpha * (outcomes[t] - p_hat[m])
    return choices, outcomes


def sample_params(model: Model, rng: np.random.Generator, kappa=None):
    """Draw plausible 'true' parameters for a simulated participant."""
    theta = [rng.uniform(0.05, 0.6),        # alpha
             rng.uniform(1.0, 8.0),         # beta
             rng.normal(0.0, 0.3)]          # bias
    if model is AU:
        theta.append(rng.uniform(0.0, 1.0) if kappa is None else kappa)
    return np.array(theta)
