"""The compiled staging pass gives the NumPy path's statistics and errors."""

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from diff_stream import did_bjs, did_callaway_santanna
from diff_stream._core import staging_kernels
from diff_stream.simulate import simulate_conditional_panel


@pytest.fixture(scope="module")
def panel():
    return simulate_conditional_panel(600, 7, seed=5).with_columns(
        (0.4 + pl.col("id") % 7).alias("w")
    )


def fit(data, enabled, function=did_bjs, fml="y ~ 1", **options):
    old = staging_kernels.ENABLED
    staging_kernels.ENABLED = enabled
    try:
        return function(
            fml, data, panel=("id", "time"), cohort="first_treat", batch_units=70, **options
        )
    finally:
        staging_kernels.ENABLED = old


def beyond_the_sample(frame):
    """A tenth of the treated units are treated only after the last period."""
    late = (pl.col("id") % 10 == 3) & (pl.col("first_treat") > 0)
    return frame.with_columns(
        pl.when(late).then(99).otherwise(pl.col("first_treat")).alias("first_treat")
    )


CASES = [
    ("plain", lambda p: p, {}),
    ("weights", lambda p: p, dict(weights="w")),
    ("anticipation", lambda p: p, dict(anticipation=1)),
    ("beyond the sample", beyond_the_sample, {}),
    ("beyond with anticipation", beyond_the_sample, dict(anticipation=1)),
    (
        "unbalanced",
        lambda p: p.filter(pl.Series(np.random.default_rng(2).random(p.height) > 0.1)),
        dict(incomplete_units="keep"),
    ),
    (
        "unbalanced and weighted",
        lambda p: p.filter(pl.Series(np.random.default_rng(2).random(p.height) > 0.1)),
        dict(incomplete_units="keep", weights="w"),
    ),
    ("covariates", lambda p: p, dict(fml="y ~ x + z")),
]


@pytest.mark.parametrize(("name", "prepare", "options"), CASES, ids=[c[0] for c in CASES])
def test_statistics_match_the_numpy_path(panel, name, prepare, options):
    data = prepare(panel)
    a = fit(data, False, **options)
    b = fit(data, True, **options)
    assert a.diagnostics["cohort_sizes"] == b.diagnostics["cohort_sizes"]
    assert a.coefnames == b.coefnames
    assert_allclose(b.beta, a.beta, rtol=1e-12, atol=1e-13)
    assert_allclose(b.vcov, a.vcov, rtol=1e-11, atol=1e-14)
    ca, cb = a.cohort_time(), b.cohort_time()
    for column in ca.columns:
        if ca[column].dtype.is_float():
            assert_allclose(cb[column].to_numpy(), ca[column].to_numpy(), rtol=1e-12)
        else:
            assert ca[column].to_list() == cb[column].to_list()


def with_duplicate(p):
    return pl.concat([p, p.head(1)])


def with_moving_cohort(p):
    return p.with_columns(
        pl.when((pl.col("id") == 5) & (pl.col("time") == 3))
        .then(4)
        .otherwise(pl.col("first_treat"))
        .alias("first_treat")
    )


def with_moving_weight(p):
    return p.with_columns(
        pl.when((pl.col("id") == 5) & (pl.col("time") == 3))
        .then(9.0)
        .otherwise(pl.col("w"))
        .alias("w")
    )


def with_bad_weight(p):
    return p.with_columns(pl.when(pl.col("id") == 5).then(-1.0).otherwise(pl.col("w")).alias("w"))


ERRORS = [
    (with_duplicate, {}),
    (with_moving_cohort, {}),
    (with_moving_weight, dict(weights="w")),
    (with_bad_weight, dict(weights="w")),
    (lambda p: p.filter(~((pl.col("id") == 5) & (pl.col("time") == 2))), {}),
]


@pytest.mark.parametrize(("break_it", "options"), ERRORS)
def test_errors_match_the_numpy_path(panel, break_it, options):
    data = break_it(panel)
    messages = []
    for enabled in (False, True):
        with pytest.raises(ValueError) as caught:
            fit(data, enabled, function=did_callaway_santanna, **options)
        messages.append(str(caught.value))
    assert messages[0] == messages[1]


def test_results_do_not_depend_on_threads(panel):
    results = []
    for threads in (1, 3, 8):
        r = fit(panel, True, weights="w", n_threads=threads)
        results.append((np.asarray(r.beta), np.asarray(r.vcov)))
    for other in results[1:]:
        for x, y in zip(results[0], other, strict=True):
            assert np.array_equal(x, y)
