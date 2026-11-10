"""The panel for the continuous-treatment references (continuous_reference.R).

Usage: python benchmarks/external/continuous_panel.py tests/data/continuous_panel.csv

2,400 units over 6 periods: 40% never treated, and three cohorts first treated in
periods 3, 4 and 5. Each treated unit's dose is drawn from 1 to 10, and every
cohort contains a unit at dose 1 and one at dose 10. R's contdid fits each
cohort-period spline with the boundary knots of that cell's doses but evaluates
the curves with the boundary knots of the dose grid (contdid issue #16), so its
curves can be compared only when every cell's doses span the same range as the
grid. The effects grow with the dose (nonlinearly) and with time since treatment,
and differ by cohort; trends are parallel.
"""

import sys

import numpy as np
import polars as pl

rng = np.random.default_rng(20261110)
N, T = 2400, 6
cohort = rng.choice([0, 3, 4, 5], N, p=[0.4, 0.2, 0.2, 0.2])
dose = np.where(cohort > 0, rng.uniform(1, 10, N), 0.0)
for g in (3, 4, 5):
    members = np.flatnonzero(cohort == g)
    dose[members[0]], dose[members[1]] = 1.0, 10.0
unit = np.repeat(np.arange(N), T)
time = np.tile(np.arange(1, T + 1), N)
g, d = cohort[unit], dose[unit]
event = time - g
response = 0.3 * d + 0.04 * d**2 - 0.002 * d**3
effect = np.where((g > 0) & (event >= 0), (1 + 0.2 * event + 0.05 * (g - 3)) * response, 0.0)
y = (
    rng.normal(0, 1, N)[unit]
    + 0.2 * time
    + rng.normal(0, 0.5, T)[time - 1]
    + effect
    + rng.normal(0, 1, N * T)
)
pl.DataFrame({"id": unit, "time": time, "first_treat": g, "dose": d, "y": y}).write_csv(sys.argv[1])
