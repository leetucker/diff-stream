"""Run from the repository root: python examples/gardner.py."""

import polars as pl

from diff_stream import did_two_stage

panel = pl.read_csv("tests/data/castle.csv", null_values="NA").with_columns(
    pl.when(pl.col("effyear").is_null())
    .then(0)
    .otherwise(pl.col("effyear") + 1)
    .cast(pl.Int64)
    .alias("first_treat")
)
kwargs = dict(panel=("state", "year"), cohort="first_treat", n_threads=2)
static = did_two_stage("l_homicide ~ 1", panel, event_study=False, **kwargs)
static.summary()
dynamic = did_two_stage("l_homicide ~ 1", panel, **kwargs)
print(dynamic.select([-3, -2, 0, 1, 2]).tidy())
print(dynamic.pretrend_test(horizons=[-3, -2]))
