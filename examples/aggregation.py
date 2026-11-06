"""Aggregation and contributor inspection on a simulated unbalanced panel."""

from diff_stream import did_callaway_santanna
from diff_stream.simulate import simulate_conditional_panel

panel = simulate_conditional_panel(1200, 8, seed=99)
panel = panel.filter(~((panel["id"] % 4 == 0) & (panel["time"] == 7)))
fit = did_callaway_santanna(
    "y ~ x + z",
    panel,
    panel=("id", "time"),
    cohort="first_treat",
    incomplete_units="keep",
    n_threads=2,
)
for kind in ["cohort", "calendar"]:
    result = fit.aggregate(kind)
    result.summary()
    print(result.contributions())

# Cohorts with exposure 0 and 1 in this particular simulation.
window = fit.aggregate(balance_window=(0, 1))
print(window.composition())
print(window.diagnostics["excluded_cohorts"])
