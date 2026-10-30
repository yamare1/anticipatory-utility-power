"""Power: how often would a study detect the savouring effect at group level?

For each design, sample size N and true savouring strength kappa, simulate
many studies and run the report's planned test (logistic regression of BET
on delay, clustered by person). Power = share of studies with p < 0.05.
kappa = 0 gives the false-positive rate, which should sit near 5%.

    python scripts/03_power.py              # run simulations (~2 min on 2 cores)
    python scripts/03_power.py --plot-only  # redraw the figure from results/power.csv
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from au.designs import DESIGNS  # noqa: E402
from au.group import delay_effect, simulate_group  # noqa: E402

N_GRID = [15, 30, 60, 120]
KAPPA_GRID = [0.0, 0.25, 0.5, 1.0]
REPS = 150


def one_cell(args):
    design, n, kappa, seed = args
    rng = np.random.default_rng(seed)
    task = DESIGNS[design]()
    hits = 0
    for _ in range(REPS):
        df = simulate_group(task, n, kappa, rng)
        _, p = delay_effect(df)
        hits += p < 0.05
    return {"design": design, "n": n, "kappa": kappa, "power": hits / REPS}


def plot(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, design in zip(axes, DESIGNS):
        sub = df[df["design"] == design]
        for k in KAPPA_GRID:
            s = sub[sub["kappa"] == k]
            ax.plot(s["n"], s["power"], marker="o", label=f"kappa = {k}")
        ax.axhline(0.8, color="grey", ls=":", lw=1)
        ax.axhline(0.05, color="grey", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xticks(N_GRID, N_GRID)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("participants")
        ax.set_title(design)
    axes[0].set_ylabel("power (p < 0.05)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Power to detect a delay effect on BET choice")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "power.png", dpi=150)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot(pd.read_csv(ROOT / "results" / "power.csv"))
        sys.exit()
    jobs = [(d, n, k, 1000 * i + 7) for i, (d, n, k) in enumerate(
        (d, n, k) for d in DESIGNS for n in N_GRID for k in KAPPA_GRID)]
    with Pool(2) as pool:
        out = pool.map(one_cell, jobs)
    df = pd.DataFrame(out)
    df.to_csv(ROOT / "results" / "power.csv", index=False)
    print(df.pivot_table(index=["design", "kappa"], columns="n", values="power").round(2))
    plot(df)
