"""A panel where every unit is eventually treated: the last-treated cohort as controls."""

from pathlib import Path

import polars as pl

from diff_stream import did_bjs, did_extended_twfe, did_two_stage

ROOT = Path(__file__).resolve().parents[1]
data = pl.read_csv(ROOT / "tests/data/richer_unbalanced_panel.csv").filter(
    pl.col("first_treat") != 0
)

for label, entry in [("BJS", did_bjs), ("Gardner", did_two_stage), ("ETWFE", did_extended_twfe)]:
    options = dict(pretrend_leads=2) if label == "BJS" else {}
    with entry(
        "y~x+z",
        data,
        panel=("id", "time"),
        cohort="first_treat",
        last_treated_controls="if_needed",
        incomplete_units="keep",
        anticipation=1,
        weights="w",
        allow_time_varying_weights=True,
        n_threads=2,
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
