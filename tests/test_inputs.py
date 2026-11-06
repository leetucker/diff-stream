"""Treatment timing given as a 0/1 column, null cohorts, and evenly spaced periods.

Each alternative input must give exactly the fit of the equivalent cohort-coded
annual panel; spaced periods must show their own labels.
"""

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from diff_stream import (
    bacon_decomp,
    did_bjs,
    did_callaway_santanna,
    did_extended_twfe,
    did_sun_abraham,
    did_two_stage,
    refit_bootstrap,
)
from diff_stream.simulate import simulate_conditional_panel

ESTIMATORS = [did_callaway_santanna, did_sun_abraham, did_bjs, did_two_stage, did_extended_twfe]
KW = dict(panel=("id", "time"), n_threads=2)
DATA = simulate_conditional_panel(400, 6, seed=21)


def with_treatment(d):
    treated = (pl.col("first_treat") > 0) & (pl.col("time") >= pl.col("first_treat"))
    return d.with_columns(treated.cast(pl.Int8).alias("treated")).drop("first_treat")


def biennial(d, start=2001):
    # periods 1..6 become 2001, 2003, ..., 2011; cohorts move with them
    return d.with_columns(
        (start + 2 * (pl.col("time") - 1)).alias("time"),
        pl.when(pl.col("first_treat") == 0)
        .then(0)
        .otherwise(start + 2 * (pl.col("first_treat") - 1))
        .alias("first_treat"),
    )


def same_fit(a, b):
    assert_allclose(a.beta, b.beta, atol=1e-11, rtol=0)
    assert_allclose(a.vcov, b.vcov, atol=1e-11, rtol=0)


@pytest.mark.parametrize("estimator", ESTIMATORS, ids=lambda f: f.__name__)
def test_treatment_column_equals_cohort(estimator):
    by_cohort = estimator("y ~ 1", DATA, cohort="first_treat", **KW)
    by_treatment = estimator("y ~ 1", with_treatment(DATA), treatment="treated", **KW)
    assert by_cohort.coefnames == by_treatment.coefnames
    same_fit(by_cohort, by_treatment)


def test_treatment_column_in_diagnostic_and_bootstrap():
    a = bacon_decomp("y ~ 1", DATA, cohort="first_treat", **KW)
    b = bacon_decomp("y ~ 1", with_treatment(DATA), treatment="treated", **KW)
    assert_allclose(a.twfe_estimate, b.twfe_estimate, atol=1e-12)
    boot = dict(estimator=did_bjs, reps=3, seed=4, aggregate="overall", **KW)
    a = refit_bootstrap("y ~ 1", DATA, cohort="first_treat", **boot)
    b = refit_bootstrap("y ~ 1", with_treatment(DATA), treatment="treated", **boot)
    assert_allclose(a.se, b.se, atol=1e-12)


def test_null_cohort_is_never_treated():
    nulls = DATA.with_columns(
        pl.when(pl.col("first_treat") == 0)
        .then(None)
        .otherwise(pl.col("first_treat"))
        .alias("first_treat")
    )
    assert nulls["first_treat"].null_count() > 0
    for estimator in (did_callaway_santanna, did_bjs):
        same_fit(
            estimator("y ~ 1", DATA, cohort="first_treat", **KW),
            estimator("y ~ 1", nulls, cohort="first_treat", **KW),
        )


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda d: d.with_columns(pl.lit(2).alias("treated")), "0 or 1"),
        (
            lambda d: d.with_columns(
                pl.when(pl.col("time") == 6).then(0).otherwise(pl.col("treated")).alias("treated")
            ),
            "absorbing",
        ),
    ],
)
def test_invalid_treatment_columns(change, match):
    with pytest.raises(ValueError, match=match):
        did_callaway_santanna("y ~ 1", change(with_treatment(DATA)), treatment="treated", **KW)


def test_exactly_one_of_cohort_and_treatment():
    for options in [{}, {"cohort": "first_treat", "treatment": "first_treat"}]:
        with pytest.raises(ValueError, match="exactly one"):
            did_callaway_santanna("y ~ 1", DATA, **options, **KW)


@pytest.mark.parametrize("estimator", ESTIMATORS, ids=lambda f: f.__name__)
def test_biennial_periods_equal_consecutive_ones(estimator):
    annual = estimator("y ~ x", DATA, cohort="first_treat", **KW)
    spaced = estimator("y ~ x", biennial(DATA), cohort="first_treat", **KW)
    same_fit(annual, spaced)
    assert spaced.diagnostics["period_step"] == 2
    assert (spaced.diagnostics["time_min"], spaced.diagnostics["time_max"]) == (2001, 2011)
    if estimator is did_two_stage:
        assert spaced.coefnames == annual.coefnames  # event times count periods
        support = spaced.support()
        assert set(support["time"].to_list()) <= set(range(2001, 2012, 2))
        return
    table = spaced.cohort_time()
    assert set(table["time"].to_list()) <= set(range(2001, 2012, 2))
    assert set(table["cohort"].to_list()) <= set(range(2001, 2012, 2))
    first = spaced.coefnames[0]
    assert str(table["cohort"][0]) in first and str(table["time"][0]) in first
    assert spaced.aggregate().coefnames == annual.aggregate().coefnames
    by_cohort = spaced.aggregate("cohort")
    same_fit(annual.aggregate("cohort"), by_cohort)
    assert all(int(n.split("::")[1]) % 2 == 1 for n in by_cohort.coefnames)
    calendar = spaced.aggregate("calendar")
    assert calendar.coefnames == [
        f"calendar::{2001 + 2 * (int(n.split('::')[1]) - 1)}"
        for n in annual.aggregate("calendar").coefnames
    ]
    contributions = calendar.contributions()
    assert set(contributions["time"].to_list()) <= set(range(2001, 2012, 2))


def test_biennial_last_treated_window_and_bootstrap():
    never = DATA.filter(pl.col("first_treat") != 0)
    with pytest.warns(Warning):
        annual = did_callaway_santanna("y ~ 1", never, cohort="first_treat", **KW)
    with pytest.warns(Warning):
        spaced = did_callaway_santanna("y ~ 1", biennial(never), cohort="first_treat", **KW)
    same_fit(annual, spaced)
    assert spaced.diagnostics["reference_cohort"] == 2001 + 2 * (
        annual.diagnostics["reference_cohort"] - 1
    )
    assert spaced.diagnostics["reference"] == "last_treated"
    boot = dict(cohort="first_treat", estimator=did_bjs, reps=3, seed=4, **KW)
    a = refit_bootstrap("y ~ 1", DATA, aggregate="event_time", **boot)
    b = refit_bootstrap("y ~ 1", biennial(DATA), aggregate="event_time", **boot)
    assert a.coefnames == b.coefnames
    assert_allclose(a.se, b.se, atol=1e-11)


def test_biennial_tables_and_errors():
    spaced = biennial(DATA)
    fit = did_callaway_santanna("y ~ x", spaced, cohort="first_treat", covariate_balance=True, **KW)
    times = set(range(2001, 2012, 2))
    assert set(fit.effective_sample_size()["time"].to_list()) <= times
    assert set(fit.covariate_balance()["time"].to_list()) <= times
    assert fit.covariate_balance(time=2011).height > 0
    bjs = did_bjs("y ~ 1", spaced, cohort="first_treat", pretrend_leads=1, **KW)
    assert set(bjs.placebos().support()["time"].to_list()) <= times
    bacon = bacon_decomp("y ~ 1", spaced, cohort="first_treat", **KW)
    assert set(bacon.comparisons()["treated_cohort"].to_list()) <= times
    off_grid = spaced.with_columns(
        pl.when(pl.col("first_treat") == 2007)
        .then(2008)
        .otherwise(pl.col("first_treat"))
        .alias("first_treat")
    )
    with pytest.raises(ValueError, match="not one of the panel's periods"):
        did_callaway_santanna("y ~ 1", off_grid, cohort="first_treat", **KW)
    uneven = spaced.filter(pl.col("time") != 2005)
    with pytest.raises(ValueError, match="evenly spaced"):
        did_callaway_santanna("y ~ 1", uneven, cohort="first_treat", **KW)
    # anticipation counts periods: one period is two years here
    a = did_sun_abraham("y ~ 1", DATA, cohort="first_treat", anticipation=1, **KW)
    b = did_sun_abraham("y ~ 1", spaced, cohort="first_treat", anticipation=1, **KW)
    same_fit(a, b)
    assert np.all(np.isfinite(b.se))
