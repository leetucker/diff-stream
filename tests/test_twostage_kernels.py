"""The compiled two-stage passes give the array path's numbers."""

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from diff_stream import did_bjs, did_sun_abraham, did_two_stage
from diff_stream._core import twostage_kernels
from diff_stream.simulate import simulate_conditional_panel


@pytest.fixture(scope="module")
def panels():
    p = simulate_conditional_panel(900, 7, seed=17).with_columns(
        (0.4 + pl.col("id") % 7).alias("w")
    )
    keep = np.random.default_rng(1).random(p.height) > 0.12
    return p, p.filter(pl.Series(keep))


def fit(function, data, enabled, **options):
    old = twostage_kernels.ENABLED
    twostage_kernels.ENABLED = enabled
    try:
        fml = options.pop("fml", "y ~ x + z")
        return function(
            fml, data, panel=("id", "time"), cohort="first_treat", batch_units=97, **options
        )
    finally:
        twostage_kernels.ENABLED = old


CASES = [
    dict(),
    dict(fml="y ~ 1"),
    dict(event_study=False),
    dict(weights="w"),
    dict(anticipation=1),
    dict(incomplete_units="keep"),
    dict(incomplete_units="keep", weights="w", fml="y ~ x"),
    dict(event_study=False, incomplete_units="keep", anticipation=1),
    dict(vcov="CRV1"),
]


@pytest.mark.parametrize("options", CASES, ids=lambda o: "-".join(f"{k}={v}" for k, v in o.items()))
def test_two_stage_matches_the_array_path(panels, options):
    data = panels[1] if options.get("incomplete_units") == "keep" else panels[0]
    a = fit(did_two_stage, data, False, **options)
    b = fit(did_two_stage, data, True, **options)
    assert a.coefnames == b.coefnames
    assert_allclose(b.beta, a.beta, rtol=1e-11, atol=1e-12)
    assert_allclose(b.vcov, a.vcov, rtol=1e-9, atol=1e-13)


@pytest.mark.parametrize("function", [did_bjs, did_sun_abraham])
@pytest.mark.parametrize("keep", [False, True])
def test_covariate_reduction_matches_the_array_path(panels, function, keep):
    options = dict(incomplete_units="keep") if keep else {}
    data = panels[1] if keep else panels[0]
    a = fit(function, data, False, **options)
    b = fit(function, data, True, **options)
    assert_allclose(b.beta, a.beta, rtol=1e-10, atol=1e-12)
    assert_allclose(b.vcov, a.vcov, rtol=1e-8, atol=1e-13)


def test_results_do_not_depend_on_threads(panels):
    results = []
    for threads in (1, 3, 8):
        r = fit(did_two_stage, panels[0], True, n_threads=threads)
        results.append((np.asarray(r.beta), np.asarray(r.vcov)))
    for other in results[1:]:
        for x, y in zip(results[0], other, strict=True):
            assert np.array_equal(x, y)
