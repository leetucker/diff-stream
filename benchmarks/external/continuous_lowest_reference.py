"""Lowest-dose references for did_continuous from diff-diff's ContinuousDiD.

Usage (in an environment with diff-diff 4.0.0 and pandas):

    python benchmarks/external/continuous_lowest_reference.py \
        tests/data/continuous_lowest_panel.csv tests/data/continuous_lowest_reference

Every unit of the panel is treated, at one date, so the comparison is the units at
the lowest dose (control_group="lowest_dose", the paper's Remark 3.1): the curves are
ATT(d) - ATT(d_L). diff-diff implements it for one cohort, without covariates. Fits,
named <dose type>_<base period>, use a cubic spline in the continuous dose (``dose``,
outcome ``y``) or the saturated regression of the discrete dose (``dose_level``,
outcome ``y_level``), with a varying or universal base period. ``reference_varying``
is the continuous fit with every dose up to 3 set to 3: diff-diff's lowest-dose group
is then the units with doses up to 3, which is did_continuous's reference_dose=3.
Each fit writes overall, curves, cells and event files.
"""

import json
import sys
import warnings
from pathlib import Path

import diff_diff as dd
import numpy as np
import pandas as pd

panel, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
data = pd.read_csv(panel)
REFERENCE = 3.0
recoded = data.assign(dose=np.maximum(data["dose"], REFERENCE))
GRID = np.linspace(1, 9.5, 18)
FITS = {
    "continuous_varying": (data, "dose", "y", "varying", GRID),
    "continuous_universal": (data, "dose", "y", "universal", GRID),
    "discrete_varying": (data, "dose_level", "y_level", "varying", None),
    "discrete_universal": (data, "dose_level", "y_level", "universal", None),
    "reference_varying": (recoded, "dose", "y", "varying", GRID[GRID > REFERENCE]),
}
for name, (frame, dose, outcome, base, grid) in FITS.items():
    kind = "discrete" if dose == "dose_level" else "continuous"
    options = dict(treatment_type=kind) | (dict(degree=3, num_knots=0) if grid is not None else {})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = dd.ContinuousDiD(
            **options, dvals=grid, control_group="lowest_dose", base_period=base
        ).fit(frame, outcome=outcome, unit="id", time="time", first_treat="first_treat", dose=dose)
        event = fit.aggregate("event_study").to_dataframe()
    pd.DataFrame(
        {
            "target": ["ATT", "ACRT"],
            "estimate": [fit.att, fit.overall_acrt],
            "se": [fit.se, fit.overall_acrt_se],
        }
    ).to_csv(out / f"overall_{name}.csv", index=False, float_format="%.17g")
    pd.DataFrame(
        {
            "dose": fit.dose_grid,
            "att": fit.dose_response_att.effects,
            "att_se": fit.dose_response_att.se,
            "acrt": fit.dose_response_acrt.effects,
            "acrt_se": fit.dose_response_acrt.se,
        }
    ).to_csv(out / f"curves_{name}.csv", index=False, float_format="%.17g")
    rows = [
        {"g": int(g), "t": int(t), "att": cell["att_glob"], "acrt": cell["acrt_glob"]}
        for (g, t), cell in sorted(fit.group_time_effects.items())
    ]
    pd.DataFrame(rows).to_csv(out / f"cells_{name}.csv", index=False, float_format="%.17g")
    event[["event_time", "att", "se"]].rename(
        columns={"event_time": "event", "att": "estimate"}
    ).to_csv(out / f"event_{name}.csv", index=False, float_format="%.17g")
versions = {"diff-diff": dd.__version__, "pandas": pd.__version__, "numpy": np.__version__}
pd.DataFrame({"package": list(versions), "version": list(versions.values())}).to_csv(
    out / "versions.csv", index=False
)
print(json.dumps(versions))
