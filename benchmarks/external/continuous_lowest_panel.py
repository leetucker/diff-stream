"""The panel for did_continuous's lowest-dose comparison (continuous_lowest_reference.py).

Usage: python benchmarks/external/continuous_lowest_panel.py tests/data/continuous_lowest_panel.csv

Units are institution-major cells observed over 6 periods, all treated in period 4,
so no unit is untreated. 30 institutions each offer 18 to 30 of 40 majors. A major's
exposure is its dose (``dose``, continuous, 0.5 to 10; every unit of the
least-exposed major is at the lowest dose) and its level (``dose_level``: 1 for the
10 least-exposed majors, then 2, 4 and 7). Untreated outcomes trend by institution,
with a time-varying covariate ``x``, and each major has a shock in each period that
all its institutions share (so standard errors clustered by major are larger).
``weight`` is a cell's size, constant over time; ``weight_t`` its size in each
period.
"""

import sys

import numpy as np
import polars as pl

rng = np.random.default_rng(20261113)
S, MAJORS, T, G = 30, 40, 6, 4
exposure = np.sort(np.r_[0.5, rng.uniform(0.6, 10, MAJORS - 1)])
level = np.select(
    [np.arange(MAJORS) < 10, np.arange(MAJORS) < 20, np.arange(MAJORS) < 30], [1.0, 2.0, 4.0], 7.0
)
pairs = [
    (s, m)
    for s in range(S)
    for m in np.sort(rng.choice(MAJORS, rng.integers(18, 31), replace=False))
]
institution = np.array([s for s, m in pairs])
major = np.array([m for s, m in pairs])
N = len(pairs)
# institutions with more exposed majors trend up more, so the comparison needs them
mix = np.array([exposure[major[institution == s]].mean() for s in range(S)])
trend = 0.1 * (mix - mix.mean())[institution]
size = rng.poisson(30, N) + 1
unit = np.repeat(np.arange(N), T)
time = np.tile(np.arange(1, T + 1), N)
d, dl = exposure[major][unit], level[major][unit]
x = rng.normal(0, 1, N)[unit] + 0.15 * time * rng.normal(0, 1, N)[unit] + rng.normal(0, 0.3, N * T)
shock = rng.normal(0, 0.4, (MAJORS, T))[major[unit], time - 1]
event = time - G
treated = event >= 0
base = (
    rng.normal(0, 1, N)[unit]
    + (0.2 + trend[unit]) * time
    + 0.4 * x * time / T
    + shock
    + rng.normal(0, 1, N * T)
)
y = base + np.where(treated, (1 + 0.2 * event) * (0.3 * d + 0.03 * d**2), 0.0)
y_level = base + np.where(treated, (1 + 0.2 * event) * (0.8 * np.sqrt(dl) + 0.1 * dl), 0.0)
pl.DataFrame(
    {
        "id": unit,
        "time": time,
        "first_treat": np.full(N * T, G),
        "region": institution[unit],
        "major": major[unit],
        "x": x,
        "dose": d,
        "dose_level": dl,
        "weight": size[unit].astype(float),
        "weight_t": (rng.poisson(size[unit]) + 1).astype(float),
        "y": y,
        "y_level": y_level,
    }
).write_csv(sys.argv[1])
