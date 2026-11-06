"""Authors' wage-panel feTR example, with optional disk-backed cell inspection."""

# ruff: noqa: I001
# Load diff-stream first so scheduler memory defaults precede the Polars import.
from pathlib import Path

from diff_stream import twfe_weights

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
data = pl.read_csv(ROOT / "tests/data/twfe_wagepan.csv", null_values="NA")

with twfe_weights(
    "lwage ~ 1",
    data,
    panel=("nr", "year"),
    treatment="union",
    keep_cells=True,
    threads=2,
    preprocess_mb=4,
) as diagnostic:
    print("TWFE coefficient:", diagnostic.twfe_estimate)
    print(diagnostic.summary())
    print(diagnostic.sensitivity())
    print(diagnostic.cells().filter(pl.col("classified_weight") < 0).collect())
