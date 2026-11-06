"""Common anticipation timing across the three estimators."""

from diff_stream import did_bjs, did_callaway_santanna, did_sun_abraham
from diff_stream.simulate import simulate_conditional_panel

panel = simulate_conditional_panel(1200, 8, seed=72)
for estimator in [did_callaway_santanna, did_sun_abraham, did_bjs]:
    formula = "y ~ x + z" if estimator is did_callaway_santanna else "y ~ 1"
    result = estimator(
        formula, panel, panel=("id", "time"), cohort="first_treat", anticipation=2, n_threads=2
    )
    try:
        print(result.cohort_time())
        print(result.aggregate(balance_window=(-2, 1)).composition())
        result.aggregate("overall").summary()
    finally:
        result.cleanup()
