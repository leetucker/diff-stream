"""Run richer SA/BJS and time-varying CS weights on the frozen synthetic sample."""

from pathlib import Path

import polars as pl

from diff_stream import diff_stream


def main():
    root = Path(__file__).resolve().parents[1]
    data = pl.read_csv(root / "tests/data/richer_unbalanced_panel.csv")
    for estimator in ["cs", "sunab", "bjs"]:
        fit = diff_stream(
            "y ~ x+z" if estimator == "cs" else "y ~ x + I(x**2) | region^time",
            data,
            estimator=estimator,
            panel=("id", "time"),
            cohort="first_treat",
            weights="w",
            allow_time_varying_weights=True,
            panel_mode="unbalanced",
            pretrends=2 if estimator == "bjs" else 0,
            threads=2,
        )
        try:
            print(estimator, "—", fit.diagnostics["unit_weight_anchor"])
            print(fit.aggregate("simple").summary())
            if estimator == "bjs":
                print(fit.placebos().tidy())
                print(fit.pretrend_test())
        finally:
            fit.cleanup()


if __name__ == "__main__":
    main()
