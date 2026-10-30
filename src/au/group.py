"""Simulate a whole sample of participants and analyse it like real data."""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from .models import AU, simulate, sample_params
from .task import Task, shuffled


def simulate_group(task: Task, n_subj: int, kappa, rng, model=AU) -> pd.DataFrame:
    """One simulated study: n_subj people, each with their own parameters and trial order."""
    rows = []
    for s in range(n_subj):
        theta = sample_params(model, rng, kappa=kappa)
        tk = shuffled(task, rng)
        c, o = simulate(theta, tk, rng)
        rows.append(pd.DataFrame({
            "subj": s, "trial": np.arange(tk.n_trials), "machine": tk.machine,
            "delay": tk.delay, "safe": tk.safe, "hi": tk.hi, "catch": tk.catch,
            "bet": c, "outcome": o,
        }))
    return pd.concat(rows, ignore_index=True)


def delay_effect(df: pd.DataFrame):
    """The analysis from the report: does P(BET) rise with delay?

    Logistic regression of BET on delay (in seconds, centred), on the
    EV-matched trials only, with standard errors that account for repeated
    choices from the same person (GEE, exchangeable correlation).
    Returns (slope per second, one-sided p-value for slope > 0).
    """
    d = df[~df["catch"]].copy()
    d["delay_c"] = d["delay"] - d["delay"].mean()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = smf.gee("bet ~ delay_c", groups="subj", data=d,
                      family=sm.families.Binomial(),
                      cov_struct=sm.cov_struct.Exchangeable()).fit()
    slope, se = res.params["delay_c"], res.bse["delay_c"]
    z = slope / se
    from scipy.stats import norm
    return float(slope), float(norm.sf(z))
