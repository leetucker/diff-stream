"""Conditional CS with reusable nuisance fits and overlap diagnostics."""

from diff_stream import cs_stream
from diff_stream.simulate import simulate_conditional_panel

panel = simulate_conditional_panel(2000, 5)
for method in ["reg", "ipw", "dr"]:
    fit = cs_stream(
        "y ~ x + z", panel, panel=("id", "time"), cohort="first_treat", est_method=method, threads=4
    )
    fit.aggregate("simple").summary()
    print("Nuisance models:", fit.diagnostics["nuisance_systems"])
    print("Shared source passes:", fit.diagnostics["nuisance_source_passes"])
