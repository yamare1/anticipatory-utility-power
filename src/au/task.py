"""The slot-machine task, read straight from the Gorilla spreadsheet.

Each trial offers SAFE (a sure amount, paid immediately) or BET (win `bet_hi`
with probability P_WIN, else $0). After a winning BET the participant sees a
win cue, waits `delay_ms`, then gets paid. Each machine has its own delay:
A = 8 s, B = 12 s, C = 16 s.

On regular trials SAFE = bet_hi / 2, so SAFE and BET have the same expected
value. On catch trials SAFE = $25 > bet_hi = $20, so SAFE is clearly better;
these check attention.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

P_WIN = 0.5          # BET pays twice the safe amount or 0, so EV-matching needs p = 0.5
SCALE = 10.0         # work in units of $10 so parameters are on a sensible scale
DATA = Path(__file__).resolve().parents[2] / "data" / "slots_master.csv"


@dataclass
class Task:
    machine: np.ndarray   # 0, 1, 2 for A, B, C
    safe: np.ndarray      # sure amount, in $10 units
    hi: np.ndarray        # BET win amount, in $10 units
    delay: np.ndarray     # cue-to-reward delay in seconds
    catch: np.ndarray     # True where SAFE > EV(BET)

    @property
    def n_trials(self) -> int:
        return len(self.machine)


def load_task(path: Path = DATA) -> Task:
    """Main trials only (practice and instructions dropped), in spreadsheet order."""
    df = pd.read_csv(path)
    df = df[df["display"] == "Trials"].copy()
    machine = df["machine_img"].str.extract(r"Machine([ABC])")[0].map({"A": 0, "B": 1, "C": 2})
    return Task(
        machine=machine.to_numpy(int),
        safe=(df["safe_amt"] / SCALE).to_numpy(float),
        hi=(df["bet_hi"] / SCALE).to_numpy(float),
        delay=(df["delay_ms"] / 1000).to_numpy(float),
        catch=(df["safe_amt"] > df["bet_hi"] * P_WIN).to_numpy(bool),
    )


def shuffled(task: Task, rng: np.random.Generator) -> Task:
    """Gorilla randomises trial order per participant (randomise = 2)."""
    idx = rng.permutation(task.n_trials)
    return Task(task.machine[idx], task.safe[idx], task.hi[idx], task.delay[idx], task.catch[idx])
