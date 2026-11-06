"""Frozen external stress references: poor overlap, weight ranges and collinearity."""

import gzip
import io
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from benchmarks._estimators import public
from benchmarks._files import read_csv
from diff_stream import did_callaway_santanna

ROOT = Path(__file__).parent / "data/numerical_stress"
REFERENCES = {r["name"]: r for r in json.loads((ROOT / "reference.json").read_text())}


@pytest.mark.parametrize("scenario", ["overlap_very_weak", "weights_wide", "collinear_border"])
@pytest.mark.parametrize("mode", ["balanced", "unbalanced"])
@pytest.mark.parametrize("method", ["reg", "ipw", "dr"])
def test_external_stress_estimates_and_joint_covariance(
    scenario, mode, method, tmp_path, request
):
    if (
        sys.platform == "darwin"
        and (scenario, mode) == ("collinear_border", "unbalanced")
        and method != "reg"
    ):
        # The propensity logit's Hessian has a condition number near 4e10 here, and
        # rounding in macOS's BLAS can leave the Newton step above the tolerance.
        request.applymarker(
            pytest.mark.xfail(
                raises=ValueError,
                strict=False,
                reason="propensity logit may not converge on macOS for this design",
            )
        )
    data = read_csv(io.BytesIO(gzip.decompress((ROOT / (scenario + ".csv.gz")).read_bytes())))
    if mode == "unbalanced":
        data = data.filter(~((pl.col("id") % 4 == 0) & pl.col("time").is_in([2, 4])))
    ref = REFERENCES[f"{mode}_{scenario}_{method}"]["fit"]
    fit = did_callaway_santanna(
        "y ~ x + z",
        data,
        panel=("id", "time"),
        cohort="first_treat",
        incomplete_units=public("incomplete_units", mode),
        weights="w",
        adjustment=public("adjustment", method),
        n_threads=2,
        propensity_trim=0.8 if scenario == "overlap_very_weak" else 0.995,
        batch_units=137,
        workdir=tmp_path,
    )
    try:
        keys = [tuple(c) for c in ref["cells"]]
        ix = [keys.index(c) for c in fit.cells]
        assert_allclose(fit.beta, np.asarray(ref["beta"])[ix], atol=2e-7, rtol=0)
        assert_allclose(fit.vcov, np.asarray(ref["vcov"])[np.ix_(ix, ix)], atol=2e-7, rtol=0)
        a = fit.aggregate("overall")
        assert_allclose(a.beta[0], ref["att"], atol=2e-7, rtol=0)
        assert_allclose(a.se[0], ref["se"], atol=2e-7, rtol=0)
        if scenario == "overlap_very_weak" and method != "reg":
            assert any(c["trimmed_controls"] for c in fit.diagnostics["comparisons"])
    finally:
        fit.cleanup()
    assert not list(tmp_path.iterdir())
