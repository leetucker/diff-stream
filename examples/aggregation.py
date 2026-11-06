"""Aggregation and contributor inspection on a simulated unbalanced panel."""

from diff_stream import cs_stream
from diff_stream.simulate import simulate_conditional_panel

panel = simulate_conditional_panel(1200, 8, seed=99)
panel = panel.filter(~((panel["id"] % 4 == 0) & (panel["time"] == 7)))
fit = cs_stream(
    "y ~ x + z",
    panel,
    panel=("id", "time"),
    cohort="first_treat",
    panel_mode="unbalanced",
    threads=2,
)
for kind in ["cohort", "calendar"]:
    result = fit.aggregate(kind)
    result.summary()
    print(result.contributions())

# Cohorts with exposure 0 and 1 in this particular simulation.
window = fit.aggregate(balance_window=(0, 1))
print(window.composition())
print(window.diagnostics["excluded_cohorts"])
