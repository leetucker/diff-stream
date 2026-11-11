"""Discrete-dose references from diff-diff's ContinuousDiD (treatment_type="discrete").

Usage (in an environment with diff-diff 4.0.0 and pandas):

    python benchmarks/external/continuous_discrete_reference.py \
        tests/data/continuous_discrete_panel.csv tests/data/continuous_discrete_reference

R's contdid does not support a discrete dose ("discrete treatment not supported
yet"), so diff-diff's saturated regression (CGBS Eq. 4.1) is the package
reference. Fits use regression adjustment without covariates, a varying base
period, and never-treated or not-yet-treated controls. For each control group:

- overall_<cg>.csv: the overall ATT and ACRT, with diff-diff's standard errors
- curves_<cg>.csv: ATT(d) and ACRT(d) at the dose levels, with standard errors
- cells_<cg>.csv: each cohort-period's ATT, ACRT, ATT(d) and ACRT(d)
- event_<cg>.csv: the event study of the binarized ATT
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
for cg in ("never_treated", "not_yet_treated"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = dd.ContinuousDiD(
            treatment_type="discrete", control_group=cg, estimation_method="reg"
        ).fit(data, outcome="y", unit="id", time="time", first_treat="first_treat", dose="dose")
        event = fit.aggregate("event_study").to_dataframe()
    pd.DataFrame(
        {
            "target": ["ATT", "ACRT"],
            "estimate": [fit.att, fit.overall_acrt],
            "se": [fit.se, fit.overall_acrt_se],
        }
    ).to_csv(out / f"overall_{cg}.csv", index=False, float_format="%.17g")
    pd.DataFrame(
        {
            "dose": fit.dose_grid,
            "att": fit.dose_response_att.effects,
            "att_se": fit.dose_response_att.se,
            "acrt": fit.dose_response_acrt.effects,
            "acrt_se": fit.dose_response_acrt.se,
        }
    ).to_csv(out / f"curves_{cg}.csv", index=False, float_format="%.17g")
    rows = []
    for (g, t), cell in sorted(fit.group_time_effects.items()):
        row = {"g": int(g), "t": int(t), "att": cell["att_glob"], "acrt": cell["acrt_glob"]}
        for j, d in enumerate(fit.dose_grid):
            row[f"att_{d:g}"], row[f"acrt_{d:g}"] = cell["att_d"][j], cell["acrt_d"][j]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / f"cells_{cg}.csv", index=False, float_format="%.17g")
    event[["event_time", "att", "se"]].rename(
        columns={"event_time": "event", "att": "estimate"}
    ).to_csv(out / f"event_{cg}.csv", index=False, float_format="%.17g")
versions = {"diff-diff": dd.__version__, "pandas": pd.__version__, "numpy": np.__version__}
pd.DataFrame({"package": list(versions), "version": list(versions.values())}).to_csv(
    out / "versions.csv", index=False
)
print(json.dumps(versions))
