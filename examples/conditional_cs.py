"""Conditional CS with reusable nuisance fits and overlap diagnostics."""

from diff_stream import did_callaway_santanna
from diff_stream.simulate import simulate_conditional_panel

panel = simulate_conditional_panel(2000, 5)
for adjustment in ["regression", "ipw", "doubly_robust"]:
    fit = did_callaway_santanna(
        "y ~ x + z",
        panel,
        panel=("id", "time"),
        cohort="first_treat",
        adjustment=adjustment,
        n_threads=4,
    )
    fit.aggregate("overall").summary()
    print("Nuisance models:", fit.diagnostics["nuisance_systems"])
    print("Shared source passes:", fit.diagnostics["nuisance_source_passes"])
