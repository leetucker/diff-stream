"""Sun–Abraham reanalysis of the frozen public Castle Doctrine panel.

Run from the source checkout. Data and reference provenance are documented in
docs/castle_replication.md. This uses adoption year, not source post's later onset.
"""

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diff_stream import sunab_stream

data = pl.read_csv(Path(__file__).resolve().parents[1] / "tests/data/castle.csv", null_values="NA")
data = data.with_columns(pl.col("effyear").fill_null(0).cast(pl.Int64).alias("first_treat"))
fit = sunab_stream("l_homicide ~ 1", data, panel=("state", "year"), cohort="first_treat", threads=2)
try:
    print(fit.aggregate().tidy())
    print(fit.aggregate("simple").tidy())
    print(fit.pretrend_test(level="dynamic", horizons=[-4, -3, -2]))
    print(fit.aggregate(balance_e=2, horizons=[0, 1, 2]).composition())
finally:
    fit.cleanup()
