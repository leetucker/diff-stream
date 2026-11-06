"""Run from an installed checkout: python examples/quickstart.py."""

from diff_stream import did_bjs, did_callaway_santanna, did_sun_abraham
from diff_stream.simulate import simulate_panel

data = simulate_panel(2000, 8)
for estimator in [did_callaway_santanna, did_sun_abraham, did_bjs]:
    fit = estimator("y ~ 1", data, panel=("id", "time"), cohort="first_treat", n_threads=4)
    fit.aggregate("overall").summary()
