"""The compiled Sun-Abraham and BJS covariate passes give the array path's numbers."""

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from diff_stream import did_bjs, did_sun_abraham
from diff_stream._core import covariate_kernels, twostage_kernels
from diff_stream.simulate import simulate_conditional_panel


@pytest.fixture(scope="module")
def panels():
    p = simulate_conditional_panel(900, 7, seed=17).with_columns(
        (0.4 + pl.col("id") % 7).alias("w")
    )
    keep = np.random.default_rng(1).random(p.height) > 0.12
    return p, p.filter(pl.Series(keep))


def fit(function, data, enabled, aggregate=None, **options):
    old = covariate_kernels.ENABLED, twostage_kernels.ENABLED
    covariate_kernels.ENABLED = twostage_kernels.ENABLED = enabled
    try:
        fml = options.pop("fml", "y ~ x + z")
        result = function(
            fml, data, panel=("id", "time"), cohort="first_treat", batch_units=97, **options
        )
        return result.aggregate(aggregate) if aggregate else result
    finally:
        covariate_kernels.ENABLED, twostage_kernels.ENABLED = old


SUNAB = [
    dict(),
    dict(fml="y ~ x"),
    dict(weights="w"),
    dict(incomplete_units="keep"),
    dict(incomplete_units="keep", weights="w", fml="y ~ x"),
    dict(anticipation=1),
    dict(vcov="CRV1"),
]
BJS = [
    dict(),
    dict(aggregate="event_time"),
    dict(aggregate="overall"),
    dict(residual_averaging="event_time", aggregate="event_time"),
    dict(residual_averaging="cohort", aggregate="cohort"),
    dict(residual_averaging="all", aggregate="overall"),
    dict(leave_out=True, residual_averaging="event_time", aggregate="event_time"),
    dict(weights="w", aggregate="event_time"),
    dict(incomplete_units="keep", aggregate="event_time"),
    dict(incomplete_units="keep", weights="w", aggregate="overall", fml="y ~ x"),
    dict(anticipation=1, aggregate="event_time"),
    dict(pretrend_leads=2, aggregate="event_time"),
    # unbalanced panels without covariates take the same compiled passes
    dict(fml="y ~ 1", incomplete_units="keep", aggregate="event_time"),
    dict(fml="y ~ 1", incomplete_units="keep", aggregate="overall", weights="w"),
    dict(
        fml="y ~ 1",
        incomplete_units="keep",
        leave_out=True,
        residual_averaging="event_time",
        aggregate="event_time",
    ),
    dict(fml="y ~ 1", incomplete_units="keep", pretrend_leads=2, aggregate="event_time"),
]


def check(function, panels, options):
    data = panels[1] if options.get("incomplete_units") == "keep" else panels[0]
    a = fit(function, data, False, **options)
    b = fit(function, data, True, **options)
    assert a.coefnames == b.coefnames
    assert_allclose(b.beta, a.beta, rtol=1e-9, atol=1e-11)
    assert_allclose(b.vcov, a.vcov, rtol=1e-8, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("options", SUNAB, ids=lambda o: "-".join(f"{k}={v}" for k, v in o.items()))
def test_sun_abraham_matches_the_array_path(panels, options):
    check(did_sun_abraham, panels, options)


@pytest.mark.parametrize("options", BJS, ids=lambda o: "-".join(f"{k}={v}" for k, v in o.items()))
def test_bjs_matches_the_array_path(panels, options):
    check(did_bjs, panels, options)


@pytest.mark.parametrize("function", [did_sun_abraham, did_bjs])
def test_results_do_not_depend_on_threads(panels, function):
    results = []
    for threads in (1, 3, 8):
        r = fit(function, panels[0], True, n_threads=threads)
        results.append((np.asarray(r.beta), np.asarray(r.vcov)))
    for other in results[1:]:
        for x, y in zip(results[0], other, strict=True):
            assert np.array_equal(x, y)
