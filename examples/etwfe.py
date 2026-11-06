"""Documented public mpdta application; run from the repository root."""

import polars as pl

from diff_stream import etwfe_stream

panel = pl.read_csv("tests/data/mpdta.csv")
fit = etwfe_stream(
    "lemp ~ lpop",
    panel,
    panel=("countyreal", "year"),
    cohort="first.treat",
    control_group="notyettreated",
    threads=2,
)
print(fit.aggregate().tidy())
print(fit.aggregate("simple").tidy())
