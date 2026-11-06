"""Sampling-weighted CS/SA/BJS, aggregation and physical sample diagnostics."""

import polars as pl

from diff_stream import bjs_stream, cs_stream, sunab_stream
from diff_stream.simulate import simulate_conditional_panel

data = simulate_conditional_panel(1200, 8, seed=72).with_columns(
    (0.2 + pl.col("id") % 9).alias("sampling_weight")
)
for estimator in (cs_stream, sunab_stream, bjs_stream):
    fit = estimator(
        "y ~ x + z" if estimator is cs_stream else "y ~ 1",
        data,
        panel=("id", "time"),
        cohort="first_treat",
        weights="sampling_weight",
        anticipation=2,
        threads=2,
    )
    try:
        print(estimator.__name__)
        print(fit.aggregate("simple").tidy())
        print(fit.group_time().select("cohort", "time", "weight_units", "weight_total"))
    finally:
        fit.cleanup()
