"""Clean pre-trend testing and CS unit-multiplier simultaneous bands."""

from diff_stream import did_callaway_santanna
from diff_stream.simulate import simulate_conditional_panel

data = simulate_conditional_panel(1200, 8, seed=72)
fit = did_callaway_santanna(
    "y ~ x + z",
    data,
    panel=("id", "time"),
    cohort="first_treat",
    anticipation=2,
    multiplier_bootstrap_reps=999,
    multiplier_bootstrap_seed=42,
    n_threads=2,
)
event_study = fit.aggregate()
print(event_study.multiplier_confint())
print(fit.pretrend_test(level="event_time"))
