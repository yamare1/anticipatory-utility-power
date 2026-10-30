"""Model recovery: if the data came from model X, does AIC / BIC pick model X?

Simulate people from EV-RL and from AU-RL, fit both models to every person,
and count which model wins. The result is a confusion matrix.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from au import AU, EV, fit, sample_params, shuffled, simulate  # noqa: E402
from au.designs import DESIGNS  # noqa: E402

N_PER_MODEL = 100
rows = []
for name, make in DESIGNS.items():
    rng = np.random.default_rng(2)
    task = make()
    for gen in (EV, AU):
        for _ in range(N_PER_MODEL):
            # AU people get a clearly non-zero savouring effect
            theta = sample_params(gen, rng, kappa=rng.uniform(0.3, 1.0) if gen is AU else None)
            tk = shuffled(task, rng)
            c, o = simulate(theta, tk, rng)
            f_ev, f_au = fit(EV, tk, c, o, rng), fit(AU, tk, c, o, rng)
            rows.append({"design": name, "generated_by": gen.name,
                         "aic_pick": "AU-RL" if f_au.aic < f_ev.aic else "EV-RL",
                         "bic_pick": "AU-RL" if f_au.bic < f_ev.bic else "EV-RL",
                         "lr_stat": 2 * (f_ev.nll - f_au.nll)})
df = pd.DataFrame(rows)
df.to_csv(ROOT / "results" / "model_recovery.csv", index=False)

fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
order = ["EV-RL", "AU-RL"]
tables = []
for ax, (name, crit) in zip(axes, [(d, c) for d in DESIGNS for c in ("aic", "bic")]):
    sub = df[df["design"] == name]
    tab = pd.crosstab(sub["generated_by"], sub[f"{crit}_pick"], normalize="index").reindex(
        index=order, columns=order, fill_value=0)
    tables.append(tab.assign(design=name, criterion=crit.upper()))
    ax.imshow(tab.values, vmin=0, vmax=1, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{tab.values[i, j]:.2f}", ha="center", va="center")
    ax.set_xticks([0, 1], order); ax.set_yticks([0, 1], order)
    ax.set_xlabel(f"picked by {crit.upper()}"); ax.set_ylabel("true model")
    ax.set_title(name, fontsize=10)
pd.concat(tables).round(3).to_csv(ROOT / "results" / "model_recovery_confusion.csv")
fig.suptitle("Model recovery (share of simulated people), one person at a time")
fig.tight_layout()
fig.savefig(ROOT / "figures" / "model_recovery.png", dpi=150)
print(pd.concat(tables).round(2))
