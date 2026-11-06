"""Run from an installed checkout: python examples/quickstart.py."""

from diff_stream import bjs_stream, cs_stream, sunab_stream
from diff_stream.simulate import simulate_panel

data = simulate_panel(2000, 8)
for estimator in [cs_stream, sunab_stream, bjs_stream]:
    fit = estimator("y ~ 1", data, panel=("id", "time"), cohort="first_treat", threads=4)
    fit.aggregate("simple").summary()
