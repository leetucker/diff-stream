"""The panel for the covariate references of did_continuous (continuous_covariates_reference.py).

Usage: python benchmarks/external/continuous_covariates_panel.py tests/data/continuous_covariates_panel.csv

2,400 units over 6 periods in 8 regions: 40% never treated, and three cohorts first
treated in periods 3, 4 and 5. Untreated outcomes trend differently by region and
with a covariate x that changes over time, so parallel trends hold only given the
region and x. Treated units' doses depend on the region (higher in regions with
steeper trends), so the unadjusted estimates are biased. ``dose`` is continuous (1 to
10, with a unit at each end in every cohort); ``dose_level`` takes the values 1, 2, 4
and 7, every value in every cohort.
"""

import sys

import numpy as np
import polars as pl

rng = np.random.default_rng(20261112)
N, T, R = 2400, 6, 8
cohort = rng.choice([0, 3, 4, 5], N, p=[0.4, 0.2, 0.2, 0.2])
region = rng.integers(0, R, N)
trend = np.linspace(-0.3, 0.4, R)[region]
dose = np.where(cohort > 0, np.clip(rng.uniform(1, 10, N) + 3 * trend, 1, 10), 0.0)
level = np.where(cohort > 0, np.array([1.0, 2.0, 4.0, 7.0])[rng.integers(0, 4, N)], 0.0)
for g in (3, 4, 5):
    members = np.flatnonzero(cohort == g)
    dose[members[0]], dose[members[1]] = 1.0, 10.0
    level[members[:4]] = [1.0, 2.0, 4.0, 7.0]
unit = np.repeat(np.arange(N), T)
time = np.tile(np.arange(1, T + 1), N)
g, d, dl = cohort[unit], dose[unit], level[unit]
x = rng.normal(0, 1, N)[unit] + 0.15 * time * rng.normal(0, 1, N)[unit] + rng.normal(0, 0.3, N * T)
event = time - g
treated = (g > 0) & (event >= 0)
y = (
    rng.normal(0, 1, N)[unit]
    + (0.2 + trend[unit]) * time
    + 0.4 * x * time / T
    + np.where(treated, (1 + 0.2 * event) * (0.3 * d + 0.03 * d**2), 0.0)
    + rng.normal(0, 1, N * T)
)
y_level = y - np.where(treated, (1 + 0.2 * event) * (0.3 * d + 0.03 * d**2), 0.0)
y_level += np.where(treated, (1 + 0.2 * event) * (0.8 * np.sqrt(dl) + 0.1 * dl), 0.0)
pl.DataFrame(
    {
        "id": unit,
        "time": time,
        "first_treat": g,
        "region": region[unit],
        "x": x,
        "dose": d,
        "dose_level": dl,
        "y": y,
        "y_level": y_level,
    }
).write_csv(sys.argv[1])
