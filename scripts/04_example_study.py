"""One simulated pilot (N = 30, moderate savouring) analysed the way the report planned.

The report's regression was  choice ~ delay + logit_prob + (1 | participant).
In the final three-machine design every BET wins with p = 0.5, so logit_prob
is constant and drops out; what's left is the delay effect.
"""
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from au.designs import DESIGNS  # noqa: E402
from au.group import delay_effect, simulate_group  # noqa: E402

N, KAPPA = 30, 0.5
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
lines = []
for ax, (name, make) in zip(axes, DESIGNS.items()):
    rng = np.random.default_rng(42)
    df = simulate_group(make(), N, KAPPA, rng)
    slope, p = delay_effect(df)

    d = df[~df["catch"]].copy()
    d["delay_c"] = d["delay"] - d["delay"].mean()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mixed = BinomialBayesMixedGLM.from_formula(
            "bet ~ delay_c", {"subj": "0 + C(subj)"}, d).fit_vb()
    m_slope = mixed.fe_mean[1]
    m_sd = mixed.fe_sd[1]

    lines.append(f"{name}: GEE slope {slope:.3f}/s (one-sided p = {p:.3f}); "
                 f"mixed-model slope {m_slope:.3f} (posterior sd {m_sd:.3f})")

    by_person = d.groupby(["subj", "delay"])["bet"].mean().reset_index()
    means = by_person.groupby("delay")["bet"].agg(["mean", "sem"])
    for s, g in by_person.groupby("subj"):
        ax.plot(g["delay"], g["bet"], color="grey", alpha=0.15, lw=0.8)
    ax.errorbar(means.index, means["mean"], yerr=1.96 * means["sem"], marker="o", lw=2, capsize=4)
    ax.set_xticks(means.index)
    ax.set_xlabel("cue-to-reward delay (s)")
    ax.set_title(f"{name}: slope p = {p:.3f}", fontsize=10)
axes[0].set_ylabel("P(BET), EV-matched trials")
fig.suptitle(f"One simulated pilot: N = {N}, true savouring kappa = {KAPPA}")
fig.tight_layout()
fig.savefig(ROOT / "figures" / "example_study.png", dpi=150)
(ROOT / "results" / "example_study.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
