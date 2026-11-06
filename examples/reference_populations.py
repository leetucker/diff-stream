"""All-treated panels: explicit last-reference windows and lead conventions."""

# ruff: noqa: I001
# Initialize scheduler defaults before Polars.
from pathlib import Path

from diff_stream import bjs_stream, etwfe_stream, gardner_stream

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
data = pl.read_csv(ROOT / "tests/data/richer_unbalanced_panel.csv").filter(
    pl.col("first_treat") != 0
)

for label, entry in [("BJS", bjs_stream), ("Gardner", gardner_stream), ("ETWFE", etwfe_stream)]:
    options = dict(pretrends=2, ess_warn_threshold=None) if label == "BJS" else {}
    with entry(
        "y~x+z",
        data,
        panel=("id", "time"),
        cohort="first_treat",
        reference_group="auto",
        panel_mode="unbalanced",
        anticipation=1,
        weights="w",
        allow_time_varying_weights=True,
        threads=2,
        preprocess_mb=4,
        batch_units=37,
        **options,
    ) as fit:
        print(
            label,
            "reference:",
            fit.diagnostics["reference_cohort"],
            "last clean date:",
            fit.diagnostics["reference_last_clean_time"],
        )
        print(fit.tidy())
        if label == "BJS":
            print(fit.placebos().tidy())
            print(fit.placebos().support())
