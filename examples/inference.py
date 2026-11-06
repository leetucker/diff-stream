"""Clean pre-trend testing and CS unit-multiplier simultaneous bands."""

from diff_stream import cs_stream
from diff_stream.simulate import simulate_conditional_panel

data = simulate_conditional_panel(1200, 8, seed=72)
fit = cs_stream(
    "y ~ x + z",
    data,
    panel=("id", "time"),
    cohort="first_treat",
    anticipation=2,
    bootstrap_reps=999,
    bootstrap_seed=42,
    threads=2,
)
event_study = fit.aggregate()
print(event_study.multiplier_confint())
print(fit.pretrend_test(level="dynamic"))
