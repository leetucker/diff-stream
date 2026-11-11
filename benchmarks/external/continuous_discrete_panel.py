"""The panel for the discrete-dose references (continuous_discrete_reference.py).

Usage: python benchmarks/external/continuous_discrete_panel.py tests/data/continuous_discrete_panel.csv

1,800 units over 6 periods: 40% never treated, and three cohorts first treated in
periods 3, 4 and 5. Each treated unit's dose is 1, 2, 4 or 7 (uneven gaps, so the
finite-difference ACRT is not a rescaled ATT), with different shares by cohort;
every cohort has units at every dose. The effects grow with the dose, nonlinearly,
and with time since treatment, and differ by cohort; trends are parallel.
"""

import sys

import numpy as np
import polars as pl

rng = np.random.default_rng(20261111)
N, T = 1800, 6
LEVELS = np.array([1.0, 2.0, 4.0, 7.0])
SHARES = {3: [0.4, 0.3, 0.2, 0.1], 4: [0.25, 0.25, 0.25, 0.25], 5: [0.1, 0.2, 0.3, 0.4]}
cohort = rng.choice([0, 3, 4, 5], N, p=[0.4, 0.2, 0.2, 0.2])
dose = np.zeros(N)
for g, p in SHARES.items():
    members = np.flatnonzero(cohort == g)
    dose[members] = rng.choice(LEVELS, len(members), p=p)
unit = np.repeat(np.arange(N), T)
time = np.tile(np.arange(1, T + 1), N)
g, d = cohort[unit], dose[unit]
event = time - g
response = 0.8 * np.sqrt(d) + 0.1 * d
effect = np.where((g > 0) & (event >= 0), (1 + 0.2 * event + 0.05 * (g - 3)) * response, 0.0)
y = (
    rng.normal(0, 1, N)[unit]
    + 0.2 * time
    + rng.normal(0, 0.5, T)[time - 1]
    + effect
    + rng.normal(0, 1, N * T)
)
pl.DataFrame({"id": unit, "time": time, "first_treat": g, "dose": d, "y": y}).write_csv(sys.argv[1])
