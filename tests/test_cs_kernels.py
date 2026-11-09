"""The compiled Callaway-Sant'Anna passes give the array path's numbers."""

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

import diff_stream.cs as cs_module
from diff_stream import did_callaway_santanna
from diff_stream._core import cs_kernels
from diff_stream.simulate import simulate_conditional_panel

KW = dict(panel=("id", "time"), cohort="first_treat", batch_units=97)


@pytest.fixture(scope="module")
def panel():
    return simulate_conditional_panel(900, 7, seed=17).with_columns(
        (0.4 + pl.col("id") % 7).alias("w"),
        (0.4 + (pl.col("id") % 7) + 0.05 * pl.col("time")).alias("wt"),
    )


def fit(panel, *, enabled=True, retain=True, **options):
    old = cs_kernels.ENABLED, cs_module.RETAIN_SHARE
    cs_kernels.ENABLED, cs_module.RETAIN_SHARE = enabled, 0.1 if retain else 0.0
    try:
        r = did_callaway_santanna(options.pop("fml", "y ~ x + z"), panel, **KW, **options)
        event = r.aggregate("event_time")
        return r, event
    finally:
        cs_kernels.ENABLED, cs_module.RETAIN_SHARE = old


CASES = [
    dict(adjustment="regression"),
    dict(adjustment="ipw"),
    dict(adjustment="doubly_robust"),
    dict(adjustment="doubly_robust", control_group="not_yet_treated"),
    dict(adjustment="ipw", control_group="not_yet_treated", base_period="universal"),
    dict(adjustment="doubly_robust", anticipation=1),
    dict(adjustment="doubly_robust", weights="w"),
    dict(adjustment="ipw", weights="w", control_group="not_yet_treated"),
    dict(adjustment="regression", weights="w", base_period="universal"),
    dict(adjustment="doubly_robust", propensity_trim=0.9),
    dict(adjustment="doubly_robust", fml="y ~ 1", weights="w"),
    dict(adjustment="ipw", multiplier_bootstrap_reps=50, multiplier_bootstrap_seed=3),
]


@pytest.mark.parametrize("options", CASES, ids=lambda o: "-".join(f"{k}={v}" for k, v in o.items()))
def test_kernels_match_the_array_path(panel, options):
    a, ea = fit(panel, enabled=False, retain=False, **options)
    b, eb = fit(panel, enabled=True, retain=True, **options)
    assert a.coefnames == b.coefnames
    assert_allclose(b.beta, a.beta, rtol=1e-11, atol=1e-12)
    assert_allclose(b.vcov, a.vcov, rtol=1e-9, atol=1e-13)
    assert_allclose(eb.beta, ea.beta, rtol=1e-11, atol=1e-12)
    assert_allclose(eb.se, ea.se, rtol=1e-9, atol=1e-13)
    for x, y in zip(a.diagnostics["comparisons"], b.diagnostics["comparisons"], strict=True):
        for key, value in x.items():
            if isinstance(value, float):
                assert y[key] == pytest.approx(value, rel=1e-9)
            else:
                assert y[key] == value, key
    if "multiplier_bootstrap_reps" in options:
        assert_allclose(b.multiplier_draws(), a.multiplier_draws(), atol=1e-9)


def test_results_do_not_depend_on_threads(panel):
    results = []
    for threads in (1, 3, 8):
        r, e = fit(panel, adjustment="doubly_robust", n_threads=threads)
        results.append((np.asarray(r.beta), np.asarray(r.vcov), np.asarray(e.se)))
    for other in results[1:]:
        for x, y in zip(results[0], other, strict=True):
            assert np.array_equal(x, y)


def test_retaining_batches_changes_nothing(panel):
    a, _ = fit(panel, retain=False, adjustment="doubly_robust")
    b, _ = fit(panel, retain=True, adjustment="doubly_robust")
    assert np.array_equal(a.beta, b.beta)
    assert np.array_equal(a.vcov, b.vcov)


def test_batches_are_retained_only_when_they_fit(panel):
    calls = []
    original = cs_module.RetainedBatches.__call__

    def counting(self):
        calls.append(self.retain)
        return original(self)

    cs_module.RetainedBatches.__call__ = counting
    try:
        fit(panel, retain=True, adjustment="ipw")
        assert any(calls) and len(calls) > 3
        calls.clear()
        fit(panel, retain=False, adjustment="ipw")
        assert not any(calls)
    finally:
        cs_module.RetainedBatches.__call__ = original
