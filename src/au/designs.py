"""The original design and a proposed redesign, so they can be compared head to head."""
import numpy as np

from .task import Task, load_task


def original() -> Task:
    """As built in Gorilla: 36 trials, 12 per machine, delays 8/12/16 s."""
    return load_task()


def redesign(n_per_machine: int = 30, delays=(2.0, 9.0, 16.0)) -> Task:
    """Wider delay range and more trials per machine.

    Keeps everything else from the original: EV-matched SAFE vs BET with the
    same stake levels, plus the same two catch trials. Wider delays make the
    savouring term change more between machines; more trials give each
    person's estimate less noise.
    """
    base = load_task()
    regular = ~base.catch
    safe_pool, hi_pool = base.safe[regular], base.hi[regular]
    machine, safe, hi, delay = [], [], [], []
    for m, d in enumerate(delays):
        idx = np.arange(n_per_machine) % len(safe_pool)
        machine += [m] * n_per_machine
        safe += list(safe_pool[idx])
        hi += list(hi_pool[idx])
        delay += [d] * n_per_machine
    # keep the catch trials, on the shortest and longest delays
    for m in (0, 2):
        machine.append(m); safe.append(2.5); hi.append(2.0); delay.append(delays[m])
    n = len(machine)
    catch = np.zeros(n, bool); catch[-2:] = True
    return Task(np.array(machine), np.array(safe), np.array(hi), np.array(delay), catch)


DESIGNS = {"original": original, "redesign": redesign}
