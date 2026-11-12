"""Sampling-weight references for did_continuous from diff-diff's ContinuousDiD.

Usage (in an environment with diff-diff 4.0.0 and pandas):

    python benchmarks/external/continuous_weights_reference.py \
        tests/data/continuous_lowest_panel.csv tests/data/continuous_covariates_panel.csv \
        tests/data/continuous_weights_reference

diff-diff takes sampling weights through a survey design (SurveyDesign(weights=...)),
without covariates; its standard errors linearize the estimates, unit by unit. Fits:

- lowest_continuous, lowest_discrete: the lowest-dose comparison on the lowest-dose
  panel, weighted by each cell's size (``weight``), with the continuous dose
  (``dose``, outcome ``y``) and the discrete dose (``dose_level``, ``y_level``);
- staggered_<comparison>: the covariate panel's three cohorts and a cubic spline in
  ``dose`` (outcome ``y``), with never-treated or not-yet-treated comparisons and
  weights of 1 + (id mod 7) / 2.

Each fit writes overall, curves, cells and event files.
"""

import json
import sys
import warnings
from pathlib import Path

import diff_diff as dd
import numpy as np
import pandas as pd

lowest, staggered, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
lowest = pd.read_csv(lowest)
staggered = pd.read_csv(staggered)
staggered["weight"] = 1 + (staggered["id"] % 7) / 2
GRID = np.linspace(1, 9.5, 18)
SPLINE = dict(treatment_type="continuous", degree=3, num_knots=0)
FITS = {
    "lowest_continuous": (lowest, "dose", "y", dict(SPLINE, control_group="lowest_dose")),
    "lowest_discrete": (
        lowest,
        "dose_level",
        "y_level",
        dict(treatment_type="discrete", control_group="lowest_dose"),
    ),
    "staggered_never_treated": (
        staggered,
        "dose",
        "y",
        dict(SPLINE, control_group="never_treated"),
    ),
    "staggered_not_yet_treated": (
        staggered,
        "dose",
        "y",
        dict(SPLINE, control_group="not_yet_treated"),
    ),
}
for name, (frame, dose, outcome, options) in FITS.items():
    grid = GRID if options["treatment_type"] == "continuous" else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = dd.ContinuousDiD(**options, dvals=grid).fit(
            frame,
            outcome=outcome,
            unit="id",
            time="time",
            first_treat="first_treat",
            dose=dose,
            survey_design=dd.SurveyDesign(weights="weight"),
        )
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
