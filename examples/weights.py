"""Sampling-weighted CS/SA/BJS, aggregation and physical sample diagnostics."""

import polars as pl

from diff_stream import did_bjs, did_callaway_santanna, did_sun_abraham
from diff_stream.simulate import simulate_conditional_panel

data = simulate_conditional_panel(1200, 8, seed=72).with_columns(
    (0.2 + pl.col("id") % 9).alias("sampling_weight")
)
for estimator in (did_callaway_santanna, did_sun_abraham, did_bjs):
    fit = estimator(
        "y ~ x + z" if estimator is did_callaway_santanna else "y ~ 1",
        data,
        panel=("id", "time"),
        cohort="first_treat",
        weights="sampling_weight",
        anticipation=2,
        n_threads=2,
    )
    try:
        print(estimator.__name__)
        print(fit.aggregate("overall").tidy())
        print(fit.cohort_time().select("cohort", "time", "weight_units", "weight_total"))
    finally:
        fit.cleanup()
