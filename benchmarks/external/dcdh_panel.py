"""The panels for the dCDH references (dcdh_reference.R).

Usage: python benchmarks/external/dcdh_panel.py tests/data/dcdh_panel.csv

Three panels of 400 groups over 8 periods, in one file (column ``panel``):

- "switching": first-period treatments 0, 1 and 2; 70% of groups switch once, up or
  down by 1 or 2, and some of those switch again, further or back to their baseline.
  Two covariates change over time; x moves with the treatment and affects y.
- "no_never": first-period treatments 0 and 1, and every group switches, so each
  baseline's last period with not-yet-switched groups ends before the panel does.
- "unbalanced": "switching" with 12% of its rows removed at random, so some groups
  are not observed just before their switch.

Effects grow with the change in treatment and with its duration; trends are parallel
within each baseline.
"""

import sys

import numpy as np
import polars as pl


def panel(seed, n, periods, never, baselines):
    rng = np.random.default_rng(seed)
    d1 = rng.choice(baselines, n)
    D = np.repeat(d1[:, None], periods, axis=1).astype(float)
    for i in range(n):
        if rng.random() < 1 - never:
            F = rng.integers(2, periods + 1)
            step = rng.choice([-1, 1]) if d1[i] > 0 else 1
            new = d1[i] + step * rng.integers(1, 3)
            D[i, F - 1 :] = new
            if rng.random() < 0.3 and F < periods:
                D[i, rng.integers(F + 1, periods + 1) - 1 :] = (
                    new + step if rng.random() < 0.5 else d1[i]
                )
    t = np.arange(1, periods + 1)
    dev = D - d1[:, None]
    y = (
        rng.normal(size=n)[:, None]
        + 0.3 * t
        + 0.2 * d1[:, None] * t
        + 0.8 * dev
        + 0.1 * np.cumsum(dev, axis=1)
        + rng.normal(size=(n, periods))
    )
    x = rng.normal(size=(n, periods)).cumsum(axis=1) + 0.5 * dev
    z = rng.normal(size=(n, periods))
    y = y + 0.7 * x - 0.3 * z
    return pl.DataFrame(
        {
            "id": np.repeat(np.arange(n), periods),
            "time": np.tile(t, n),
            "d": D.ravel(),
            "x": x.ravel(),
            "z": z.ravel(),
            "y": y.ravel(),
        }
    )


switching = panel(20261110, 400, 8, 0.3, [0, 1, 2])
keep = np.random.default_rng(20261112).random(len(switching)) > 0.12
frames = [
    switching.with_columns(pl.lit("switching").alias("panel")),
    panel(20261111, 400, 8, 0.0, [0, 1]).with_columns(pl.lit("no_never").alias("panel")),
    switching.filter(pl.Series(keep)).with_columns(pl.lit("unbalanced").alias("panel")),
]
pl.concat(frames).write_csv(sys.argv[1])
