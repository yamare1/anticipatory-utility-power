"""Parameter recovery: simulate people with known parameters, fit, compare.

If a parameter can't be recovered from simulated data where we know the
answer, it can't be trusted when fitted to real data either. Run for the
original Gorilla design and for the proposed redesign.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from au import AU, fit, sample_params, shuffled, simulate  # noqa: E402
from au.designs import DESIGNS  # noqa: E402

N_SUBJ = 200
summary = {}
fig, axes = plt.subplots(2, 4, figsize=(13, 6.4))

for row, (name, make) in enumerate(DESIGNS.items()):
    rng = np.random.default_rng(1)
    task = make()
    recs = []
    for _ in range(N_SUBJ):
        true = sample_params(AU, rng)
        tk = shuffled(task, rng)
        c, o = simulate(true, tk, rng)
        est = fit(AU, tk, c, o, rng).theta
        recs.append({**{f"true_{p}": v for p, v in zip(AU.params, true)},
                     **{f"est_{p}": v for p, v in zip(AU.params, est)}})
    df = pd.DataFrame(recs)
    df.to_csv(ROOT / "results" / f"parameter_recovery_{name}.csv", index=False)
    summary[name] = {p: np.corrcoef(df[f"true_{p}"], df[f"est_{p}"])[0, 1] for p in AU.params}
    for ax, p in zip(axes[row], AU.params):
        ax.scatter(df[f"true_{p}"], df[f"est_{p}"], s=8, alpha=0.5)
        lo, hi = df[f"true_{p}"].min(), df[f"true_{p}"].max()
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(f"{name}: {p}  (r = {summary[name][p]:.2f})", fontsize=9)
        ax.set_xlabel("true"); ax.set_ylabel("recovered")
    print(name, {k: round(v, 2) for k, v in summary[name].items()})

pd.DataFrame(summary).round(3).to_csv(ROOT / "results" / "parameter_recovery_r.csv")

# Robust metrics: rank correlation, median absolute error, share of fits stuck at a bound
from scipy.stats import spearmanr  # noqa: E402
lines = []
for name in DESIGNS:
    df = pd.read_csv(ROOT / "results" / f"parameter_recovery_{name}.csv")
    for p, (lo, hi) in zip(AU.params, AU.bounds):
        est = df[f"est_{p}"]
        lines.append({"design": name, "param": p,
                      "spearman": spearmanr(df[f"true_{p}"], est)[0],
                      "median_abs_error": (df[f"true_{p}"] - est).abs().median(),
                      "share_at_bound": ((est <= lo + 1e-6) | (est >= hi - 1e-6)).mean()})
pd.DataFrame(lines).round(3).to_csv(ROOT / "results" / "parameter_recovery_metrics.csv", index=False)
fig.suptitle("Parameter recovery, AU-RL: original (36 trials, 8/12/16 s) vs redesign (92 trials, 2/9/16 s)")
fig.tight_layout()
fig.savefig(ROOT / "figures" / "parameter_recovery.png", dpi=150)
