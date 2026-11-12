"""The panels for did_dcdh's continuous first-period treatment (dcdh_extensions_reference.R).

Usage: python benchmarks/external/dcdh_continuous_panel.py tests/data/dcdh_continuous_panel.csv

Two panels of 400 groups over 8 periods, in one file (column ``panel``):

- "continuous": each group's first-period treatment is drawn from a continuous
  distribution (0.5 to 3), so no two groups share one. 70% of groups switch once, up
  or down by 0.5 to 1.5, and some of those switch again, further or back to their
  first-period treatment. Untreated outcomes trend with a quadratic in the
  first-period treatment, so parallel trends hold only given it; a covariate x
  changes over time and affects y.
- "continuous_unbalanced": "continuous" with 12% of its rows removed at random.

Effects grow with the change in treatment and with its duration. ``w`` is a weight
that varies over time, ``region`` one of three, and ``cluster`` a group of 8 groups.
"""

import sys

import numpy as np
import polars as pl

rng = np.random.default_rng(20261114)
N, T = 400, 8
d1 = rng.uniform(0.5, 3.0, N)
D = np.repeat(d1[:, None], T, axis=1)
for i in range(N):
    if rng.random() < 0.7:
        F = rng.integers(2, T + 1)
        step = rng.choice([-1.0, 1.0])
        new = d1[i] + step * rng.uniform(0.5, 1.5)
        D[i, F - 1 :] = new
        if rng.random() < 0.3 and F < T:
            D[i, rng.integers(F + 1, T + 1) - 1 :] = (
                new + step * 0.5 if rng.random() < 0.5 else d1[i]
            )
t = np.arange(1, T + 1)[None, :]
x = rng.normal(0, 1, (N, T)) + 0.1 * t
duration = np.cumsum(D != d1[:, None], axis=1)
y = (
    rng.normal(0, 1, N)[:, None]
    + (0.2 + 0.3 * d1 - 0.08 * d1**2)[:, None] * t
    + 0.5 * x
    + (1 + 0.1 * duration) * (D - d1[:, None])
    + rng.normal(0, 1, (N, T))
)
w = rng.uniform(0.5, 2.0, (N, T))
region = rng.integers(0, 3, N)
frame = pl.DataFrame(
    {
        "id": np.repeat(np.arange(N), T),
        "time": np.tile(np.arange(1, T + 1), N),
        "d": D.ravel(),
        "y": y.ravel(),
        "x": x.ravel(),
        "w": w.ravel(),
        "region": np.repeat(region, T),
        "cluster": np.repeat(np.arange(N) // 8, T),
    }
)
missing = rng.random(frame.height) < 0.12
pl.concat(
    [
        frame.with_columns(panel=pl.lit("continuous")),
        frame.filter(pl.Series(~missing)).with_columns(panel=pl.lit("continuous_unbalanced")),
    ]
).write_csv(sys.argv[1])
