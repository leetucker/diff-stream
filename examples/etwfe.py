"""Documented public mpdta application; run from the repository root."""

import polars as pl

from diff_stream import did_extended_twfe

panel = pl.read_csv("tests/data/mpdta.csv")
fit = did_extended_twfe(
    "lemp ~ lpop",
    panel,
    panel=("countyreal", "year"),
    cohort="first.treat",
    control_group="not_yet_treated",
    n_threads=2,
)
print(fit.aggregate().tidy())
print(fit.aggregate("overall").tidy())
