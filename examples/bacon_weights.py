"""Weighted TWFE decomposition with physical counts and sampling-weight totals."""

import polars as pl

from diff_stream import bacon_decomp
from diff_stream.simulate import simulate_panel

data = simulate_panel(2000, 8, seed=72).with_columns(
    (0.2 + pl.col("id") % 9).alias("sampling_weight")
)
bacon = bacon_decomp(
    "y ~ 1",
    data,
    panel=("id", "time"),
    cohort="first_treat",
    weights="sampling_weight",
    threads=2,
)
print("Weighted TWFE:", bacon.twfe_estimate)
print("Reconstructed:", bacon.decomposed_estimate)
print(bacon.summary())
print(
    bacon.comparisons().select(
        "treated_cohort",
        "control_cohort",
        "n_treated_units",
        "treated_weight_total",
        "estimate",
        "weight",
        "contribution",
    )
)
