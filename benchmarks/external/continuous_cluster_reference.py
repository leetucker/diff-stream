"""Clustered references for did_continuous from diff-diff's ContinuousDiD.

Usage (in an environment with diff-diff 4.0.0 and pandas):

    python benchmarks/external/continuous_cluster_reference.py \
        tests/data/continuous_lowest_panel.csv tests/data/continuous_covariates_panel.csv \
        tests/data/continuous_cluster_reference

diff-diff clusters standard errors through a survey design's primary sampling units
(SurveyDesign(psu=...)), without weights here; its variance is G/(G-1) times the sum of
the clusters' squared score sums. Fits:

- reference_continuous: the lowest-dose panel's continuous dose with every dose up to 3
  set to 3 (did_continuous's reference_dose=3), clustered by major (the reference
  units span 12 majors);
- lowest_discrete: the discrete dose, whose lowest level spans 10 majors, by major;
- staggered_<comparison>: the covariate panel's three cohorts and a cubic spline in
  ``dose``, never-treated or not-yet-treated comparisons, clustered by region.

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
recoded = lowest.assign(dose=np.maximum(lowest["dose"], 3.0))
GRID = np.linspace(1, 9.5, 18)
SPLINE = dict(treatment_type="continuous", degree=3, num_knots=0)
FITS = {
    "reference_continuous": (
        recoded,
        "dose",
        "y",
        "major",
        dict(SPLINE, control_group="lowest_dose"),
        GRID[GRID > 3],
    ),
    "lowest_discrete": (
        lowest,
        "dose_level",
        "y_level",
        "major",
        dict(treatment_type="discrete", control_group="lowest_dose"),
        None,
    ),
    "staggered_never_treated": (
        staggered,
        "dose",
        "y",
        "region",
        dict(SPLINE, control_group="never_treated"),
        GRID,
    ),
    "staggered_not_yet_treated": (
        staggered,
        "dose",
        "y",
        "region",
        dict(SPLINE, control_group="not_yet_treated"),
        GRID,
    ),
}
for name, (frame, dose, outcome, cluster, options, grid) in FITS.items():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = dd.ContinuousDiD(**options, dvals=grid).fit(
            frame,
            outcome=outcome,
            unit="id",
            time="time",
            first_treat="first_treat",
            dose=dose,
            survey_design=dd.SurveyDesign(psu=cluster),
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
