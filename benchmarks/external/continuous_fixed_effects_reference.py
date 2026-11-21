"""References for did_continuous with two fixed-effect terms, from diff-diff's ContinuousDiD.

Usage (in an environment with diff-diff 4.0.0 and pandas):

    python benchmarks/external/continuous_fixed_effects_reference.py \
        tests/data/continuous_covariates_panel.csv tests/data/continuous_fixed_effects_reference

diff-diff absorbs no fixed effects. Additive fixed effects are the same regression as
their levels entered as dummies (one level of each term left out, with the intercept),
so the reference regresses the comparison units' change on x and the dummies of region
and of a unit-level industry, (id * 7 + 3) mod 5, which is not nested in region; the
tests add the same industry and absorb both terms (``y ~ x | region + industry``).
diff-diff's regression adjustment (estimation_method="reg") takes the covariates at the
base period. Fits use a varying base period and never-treated or not-yet-treated
comparisons, with a cubic spline in the continuous dose (``dose``, outcome ``y``) and the
saturated regression of the discrete dose (``dose_level``, outcome ``y_level``). Each fit
writes overall, curves, cells and event files, named <dose type>_<comparison>.
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
data["industry"] = (data["id"] * 7 + 3) % 5
covariates = ["x"]
for term in ("region", "industry"):
    for level in sorted(data[term].unique())[1:]:
        name = f"{term}_{level}"
        data[name] = (data[term] == level).astype(float)
        covariates.append(name)
FITS = {
    "continuous": dict(treatment_type="continuous", degree=3, num_knots=0),
    "discrete": dict(treatment_type="discrete"),
}
GRID = np.linspace(1, 10, 19)
for kind, options in FITS.items():
    dose, outcome = ("dose", "y") if kind == "continuous" else ("dose_level", "y_level")
    for cg in ("never_treated", "not_yet_treated"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = dd.ContinuousDiD(
                **options,
                dvals=GRID if kind == "continuous" else None,
                control_group=cg,
                estimation_method="reg",
            ).fit(
                data,
                outcome=outcome,
                unit="id",
                time="time",
                first_treat="first_treat",
                dose=dose,
                covariates=covariates,
            )
            event = fit.aggregate("event_study").to_dataframe()
        name = f"{kind}_{cg}"
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
