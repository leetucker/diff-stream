"""Separate BJS clean leads, treatment effects and a joint pre-trend test."""

from diff_stream import bjs_stream
from diff_stream.simulate import simulate_panel

data = simulate_panel(1200, 8, seed=72)
fit = bjs_stream(
    "y ~ 1",
    data,
    panel=("id", "time"),
    cohort="first_treat",
    pretrends=1,
    threads=2,
)
try:
    print(fit.placebos().tidy())
    print(fit.pretrend_test())
    print(fit.aggregate("simple").tidy())
finally:
    fit.cleanup()
print(fit.placebos().support())
