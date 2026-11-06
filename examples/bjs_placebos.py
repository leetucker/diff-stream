"""Separate BJS clean leads, treatment effects and a joint pre-trend test."""

from diff_stream import did_bjs
from diff_stream.simulate import simulate_panel

data = simulate_panel(1200, 8, seed=72)
fit = did_bjs(
    "y ~ 1",
    data,
    panel=("id", "time"),
    cohort="first_treat",
    pretrend_leads=1,
    n_threads=2,
)
try:
    print(fit.placebos().tidy())
    print(fit.pretrend_test())
    print(fit.aggregate("overall").tidy())
finally:
    fit.cleanup()
print(fit.placebos().support())
