"""Sun–Abraham reanalysis of the frozen public Castle Doctrine panel.

Treatment starts in the adoption year (effyear); states that never adopted
have a missing effyear, coded 0.
"""

from pathlib import Path

import polars as pl

from diff_stream import did_sun_abraham

data = pl.read_csv(Path(__file__).resolve().parents[1] / "tests/data/castle.csv", null_values="NA")
data = data.with_columns(pl.col("effyear").fill_null(0).cast(pl.Int64).alias("first_treat"))
fit = did_sun_abraham(
    "l_homicide ~ 1", data, panel=("state", "year"), cohort="first_treat", n_threads=2
)
try:
    print(fit.aggregate().tidy())
    print(fit.aggregate("overall").tidy())
    print(fit.pretrend_test(level="event_time", horizons=[-4, -3, -2]))
    print(fit.aggregate(balanced_horizon=2, horizons=[0, 1, 2]).composition())
finally:
    fit.cleanup()
