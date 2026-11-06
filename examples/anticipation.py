"""Common anticipation timing across the three estimators."""

from diff_stream import bjs_stream, cs_stream, sunab_stream
from diff_stream.simulate import simulate_conditional_panel

panel = simulate_conditional_panel(1200, 8, seed=72)
for estimator in [cs_stream, sunab_stream, bjs_stream]:
    formula = "y ~ x + z" if estimator is cs_stream else "y ~ 1"
    result = estimator(
        formula, panel, panel=("id", "time"), cohort="first_treat", anticipation=2, threads=2
    )
    try:
        print(result.group_time())
        print(result.aggregate(balance_window=(-2, 1)).composition())
        result.aggregate("simple").summary()
    finally:
        result.cleanup()
