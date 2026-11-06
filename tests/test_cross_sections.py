"""Callaway-Sant'Anna on repeated cross sections against R did (panel=FALSE).

The tolerance is the R comparisons' (2e-7): R's propensity logit stops at its
own convergence tolerance.
"""

from itertools import product
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from benchmarks._files import read_csv
from diff_stream import did_callaway_santanna

DATA = Path(__file__).parent / "data"
REF = DATA / "rcs_reference"
SAMPLE = read_csv(DATA / "rcs_sample.csv")
CASES = list(
    product(["nevertreated", "notyettreated"], ["varying", "universal"], ["reg", "ipw", "dr"], "uw")
)
PUBLIC = {
    "nevertreated": "never_treated",
    "notyettreated": "not_yet_treated",
    "reg": "regression",
    "ipw": "ipw",
    "dr": "doubly_robust",
}


@pytest.mark.parametrize("control,base,method,weighted", CASES)
def test_matches_r_did_repeated_cross_sections(control, base, method, weighted):
    prefix = f"rcs_{control}_{base}_{method}_{weighted}"
    fit = did_callaway_santanna(
        "y ~ x + z",
        SAMPLE,
        time="time",
        cohort="first_treat",
        control_group=PUBLIC[control],
        base_period=base,
        adjustment=PUBLIC[method],
        weights="w" if weighted == "w" else None,
        effective_sample_warning=None,
        n_threads=2,
    )
    ref = read_csv(REF / f"{prefix}.csv", null_values="NA")
    keys = list(zip(ref["g"].to_list(), ref["t"].to_list(), strict=True))
    ix = [keys.index(c) for c in fit.cells]
    assert len(ix) == ref.filter(pl.col("se").is_not_null() & pl.col("se").is_not_nan()).height
    assert_allclose(fit.beta, ref["att"].to_numpy()[ix], atol=2e-7, rtol=0)
    V = read_csv(REF / f"{prefix}_vcov.csv").to_numpy()
    assert_allclose(fit.vcov, V[np.ix_(ix, ix)], atol=2e-7, rtol=0)
    for kind, ours in [
        ("dynamic", "event_time"),
        ("simple", "overall"),
        ("group", "cohort"),
        ("calendar", "calendar"),
    ]:
        expected = read_csv(REF / f"{prefix}_{kind}.csv", null_values="NA")
        result = fit.aggregate(ours)
        if kind != "simple":
            labels = [int(n.split("::")[1]) for n in result.coefnames]
            expected = expected.filter(pl.col("event").is_in(labels)).sort("event")
            assert labels == sorted(labels) and len(labels) == expected.height
        assert_allclose(result.beta, expected["att"].to_numpy(), atol=2e-7, rtol=0)
        assert_allclose(result.se, expected["se"].to_numpy(), atol=2e-7, rtol=0)
    assert fit.diagnostics["repeated_cross_sections"] is True
    assert fit.n_clusters == {"observation": SAMPLE.height}


def test_cross_section_options():
    with pytest.raises(ValueError, match="not both"):
        did_callaway_santanna(
            "y ~ 1", SAMPLE, panel=("id", "time"), time="time", cohort="first_treat"
        )
    with pytest.raises(ValueError, match="time= for repeated cross sections"):
        did_callaway_santanna("y ~ 1", SAMPLE, cohort="first_treat")
    with pytest.raises(ValueError, match="does not apply"):
        did_callaway_santanna(
            "y ~ 1", SAMPLE, time="time", cohort="first_treat", incomplete_units="drop"
        )
    # a column called "observation" in the data does not clash with the row index
    renamed = SAMPLE.with_columns(pl.col("x").alias("observation"))
    a = did_callaway_santanna("y ~ 1", SAMPLE, time="time", cohort="first_treat", n_threads=2)
    b = did_callaway_santanna("y ~ 1", renamed, time="time", cohort="first_treat", n_threads=2)
    assert_allclose(a.beta, b.beta, atol=1e-12)
