"""The panel for the absorbed fixed-effect references (absorbed_reference.R).

Usage: python benchmarks/external/absorbed_panel.py tests/data/absorbed_panel.csv

1,200 units in 150 counties over 8 periods, so county-by-period effects have
1,200 levels: more than a dense design takes. Industries cross counties.
Outcomes carry county-period shocks, industry trends, a covariate, and
effects that vary by cohort and event time. Every county has never-treated
units.
"""

import sys

import numpy as np
import polars as pl

rng = np.random.default_rng(20261106)
N, T = 1200, 8
unit = np.repeat(np.arange(N), T)
time = np.tile(np.arange(1, T + 1), N)
county = rng.integers(0, 150, N)
industry = rng.integers(0, 12, N)
first_treat = rng.choice([0, 3, 4, 5, 6, 7], N, p=[0.3, 0.14, 0.14, 0.14, 0.14, 0.14])
# Two never-treated units in every county, so that every county-period has
# untreated observations and every treated outcome can be imputed.
for c in range(150):
    first_treat[np.flatnonzero(county == c)[:2]] = 0
shock = rng.normal(0, 1, (150, T))
g = first_treat[unit]
x = rng.normal(0, 1, N * T) + 0.2 * time
event = time - g
effect = np.where((g > 0) & (event >= 0), 1 + 0.25 * event + 0.1 * g, 0.0)
y = (
    rng.normal(0, 1, N)[unit]
    + 0.1 * time
    + shock[county[unit], time - 1]
    + 0.05 * industry[unit] * time
    + 0.5 * x
    + effect
    + rng.normal(0, 1, N * T)
)
pl.DataFrame(
    {
        "id": unit,
        "time": time,
        "first_treat": g,
        "county": county[unit],
        "industry": industry[unit],
        "x": x,
        "w": (0.5 + (unit % 5) / 4),
        "y": y,
    }
).write_csv(sys.argv[1])
