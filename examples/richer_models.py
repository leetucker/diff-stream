"""Run richer SA/BJS and time-varying CS weights on the frozen synthetic sample."""

from pathlib import Path

import polars as pl

from diff_stream import did_bjs, did_callaway_santanna, did_sun_abraham


def main():
    root = Path(__file__).resolve().parents[1]
    data = pl.read_csv(root / "tests/data/richer_unbalanced_panel.csv")
    common = dict(
        panel=("id", "time"),
        cohort="first_treat",
        weights="w",
        allow_time_varying_weights=True,
        incomplete_units="keep",
        n_threads=2,
    )
    fits = [
        ("CS", did_callaway_santanna, "y ~ x+z", {}),
        ("SA", did_sun_abraham, "y ~ x + I(x**2) | region^time", {}),
        ("BJS", did_bjs, "y ~ x + I(x**2) | region^time", {"pretrend_leads": 2}),
    ]
    for label, estimator, fml, options in fits:
        fit = estimator(fml, data, **common, **options)
        try:
            print(label, "—", fit.diagnostics["unit_weight_anchor"])
            print(fit.aggregate("overall").summary())
            if label == "BJS":
                print(fit.placebos().tidy())
                print(fit.pretrend_test())
        finally:
            fit.cleanup()


if __name__ == "__main__":
    main()
