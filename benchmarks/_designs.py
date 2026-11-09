"""The simulated panels and the scenarios of the three benchmark suites.

A *design* is a panel: n units over T periods, a set of adoption cohorts, and the
columns the suite needs. A *scenario* is a letter naming a design and the
estimators and libraries compared on it. Letters are unique across the suites
(``bench_ec2.py launch --scenarios A C H`` picks the suites that define them):

    scale_benchmark.py        A  8 periods, 3 cohorts          the event study of
                              B  8 periods, 7 cohorts          every estimator,
                              C  30 periods, 3 cohorts         no covariates
                              D  30 periods, 29 cohorts
                              E  8 periods, 7 cohorts, 10% of the rows missing
                              F  8 periods, 7 cohorts          the diagnostics (Goodman-Bacon,
                                                               TWFE weights)
    covariates_benchmark.py   G  two covariates (selection on them): Callaway-Sant'Anna,
                                 Sun-Abraham, BJS, two-stage, extended TWFE
                              H  region-by-period fixed effects (240 levels) and a covariate
    absorbed_benchmark.py     I  county-by-period fixed effects (n/40 counties: millions of levels)
                              J  county-by-period and industry-by-period, and a covariate

The data of a size are written once, in chunks of units, by `data_path`: chunk c is
seeded SEED + c, so a size's data never change, and a panel of 300 million rows
never has to fit in memory. Every file has id, time, first_treat (0 = never
treated) and y; the covariate and fixed-effect columns are in the designs that
use them.

The outcome is unit effect + common trend + a trend in the unit's covariate
(parallel trends hold conditional on it) + the treatment effect + noise. The
effect of cohort g at event time e >= 0 is 1 + 0.2 e + 0.05 g: it grows over time
and differs across cohorts, which is what the estimators exist for. Units are
never treated with probability 0.3; with covariates, the probability of
adoption rises with the unit's x.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SEED = 20261106
CHUNK = 250_000  # units per chunk
ROW_GROUP = 1_000_000  # rows per Parquet row group
NEVER = 0.3  # share of never-treated units
UNITS_PER_COUNTY = 40
REGIONS = 30
INDUSTRIES = 12


@dataclass(frozen=True)
class Design:
    periods: int
    cohorts: str  # "sparse" (3 cohorts) or "dense" (every period from the second)
    columns: str = "plain"  # "plain", "covariates" or "counties"
    missing: float = 0.0  # share of the rows (outside the first period) left out
    # the file's name; designs with equal data share a file
    key: str = field(init=False)

    def __post_init__(self):
        k = f"T{self.periods}-{self.cohorts}-{self.columns}"
        object.__setattr__(
            self, "key", k + (f"-miss{int(self.missing * 100)}" if self.missing else "")
        )


# scenario -> design
DESIGNS = {
    "A": Design(8, "sparse"),
    "B": Design(8, "dense"),
    "C": Design(30, "sparse"),
    "D": Design(30, "dense"),
    "E": Design(8, "dense", missing=0.1),
    "F": Design(8, "dense"),
    "G": Design(8, "dense", "covariates"),
    "H": Design(8, "dense", "covariates"),
    "I": Design(8, "dense", "counties"),
    "J": Design(8, "dense", "counties"),
}


def cohort_periods(design):
    """The adoption periods of the treated cohorts."""
    T = design.periods
    if design.cohorts == "sparse":
        return np.array([int(np.ceil(T * f)) for f in (0.3, 0.5, 0.75)])
    return np.arange(2, T + 1)


def shock_tables(design, n):
    """The period shocks shared by units in a group (region, county, industry), the
    same for every chunk of a size: arrays (groups, T)."""
    rng = np.random.default_rng(SEED - 1)
    T = design.periods
    if design.columns == "covariates":
        return {"region": rng.normal(0, 0.5, (REGIONS, T))}
    if design.columns == "counties":
        return {
            "county": rng.normal(0, 0.5, (n_counties(n), T)),
            "industry": rng.normal(0, 0.3, (INDUSTRIES, T)),
        }
    return {}


def n_counties(n):
    return max(10, -(-n // UNITS_PER_COUNTY))


def simulate_chunk(design, n, start, count, seed, tables):
    """Rows of units start, ..., start+count-1 (of n) as a pyarrow table."""
    import pyarrow as pa

    rng = np.random.default_rng(seed)
    T = design.periods
    ids = np.arange(start, start + count, dtype=np.int64)
    cohorts = cohort_periods(design)
    x_unit = rng.normal(0, 1, count)
    z_unit = rng.normal(0, 1, count)
    # selection into treatment on x and z (covariate designs), else none
    never = rng.random(count) < NEVER
    if design.columns == "covariates":
        weights = np.exp(
            0.4 * x_unit[:, None] * np.linspace(-1, 1, len(cohorts)) + 0.3 * z_unit[:, None]
        )
        weights /= weights.sum(axis=1, keepdims=True)
        pick = (rng.random(count)[:, None] > np.cumsum(weights, axis=1)).sum(axis=1)
        never = never | (rng.random(count) < 0.3 * (1 / (1 + np.exp(0.5 * x_unit))))
    else:
        pick = rng.integers(0, len(cohorts), count)
    first = np.where(never, 0, cohorts[np.minimum(pick, len(cohorts) - 1)]).astype(np.int64)
    time = np.arange(1, T + 1, dtype=np.int64)
    e = time[None, :] - first[:, None]
    effect = np.where((first[:, None] > 0) & (e >= 0), 1.0 + 0.2 * e + 0.05 * first[:, None], 0.0)
    y = (
        rng.normal(0, 1, count)[:, None]
        + 0.1 * time[None, :]
        + 0.3 * x_unit[:, None] * time[None, :] / T
        + effect
        + rng.normal(0, 1, (count, T))
    )
    columns = {
        "id": np.repeat(ids, T),
        "time": np.tile(time, count),
        "first_treat": np.repeat(first, T),
    }
    if design.columns == "covariates":
        region = ids % REGIONS
        y += tables["region"][region]
        columns["x"] = (x_unit[:, None] + 0.5 * rng.normal(0, 1, (count, T))).ravel()
        columns["z"] = (z_unit[:, None] + 0.5 * rng.normal(0, 1, (count, T))).ravel()
        columns["region"] = np.repeat(region, T).astype(np.int64)
    if design.columns == "counties":
        county = rng.integers(0, n_counties(n), count)
        industry = rng.integers(0, INDUSTRIES, count)
        y += (
            tables["county"][county]
            + tables["industry"][industry]
            + 0.05 * industry[:, None] * time / T
        )
        columns["x"] = (x_unit[:, None] + 0.5 * rng.normal(0, 1, (count, T))).ravel()
        columns["county"] = np.repeat(county, T).astype(np.int64)
        columns["industry"] = np.repeat(industry, T).astype(np.int64)
    columns["y"] = y.ravel()
    order = [
        "id",
        "time",
        "first_treat",
        *[c for c in columns if c not in ("id", "time", "first_treat", "y")],
        "y",
    ]
    if design.missing:
        keep = (rng.random(count * T) >= design.missing) | (columns["time"] == 1)
        columns = {k: v[keep] for k, v in columns.items()}
    return pa.table({k: columns[k] for k in order})


def data_path(design, n, data_dir):
    """The Parquet file of `n` units of `design`, written on first use."""
    import pyarrow.parquet as pq

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{design.key}_{n}.parquet"
    if path.exists():
        return path
    part = path.with_name(path.name + ".part")
    tables = shock_tables(design, n)
    writer = None
    for c, start in enumerate(range(0, n, CHUNK)):
        table = simulate_chunk(design, n, start, min(CHUNK, n - start), SEED + c, tables)
        if writer is None:
            writer = pq.ParquetWriter(part, table.schema)
        writer.write_table(table, row_group_size=ROW_GROUP)
    writer.close()
    part.replace(path)
    return path
