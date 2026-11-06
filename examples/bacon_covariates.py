"""Public bacondecomp Castle example, including covariate-adjusted components."""

from pathlib import Path

import polars as pl

from diff_stream import bacon_decomp

ROOT = Path(__file__).resolve().parents[1]
data = pl.read_csv(ROOT / "tests/data/bacon_castle_covariates.csv")
result = bacon_decomp(
    "l_homicide ~ l_pop + l_income",
    data,
    panel=("state", "year"),
    cohort="first_treat",
    n_threads=2,
)
print("TWFE:", result.twfe_estimate)
print(result.summary())
print(result.within_timing())
print(result.comparisons())
print("Reconstruction error:", result.reconstruction_error)
