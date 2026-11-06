"""CS low-support diagnostics and suppression preserve the original estimand."""

import warnings

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from diff_stream import LowEffectiveSampleWarning, cs_stream, sunab_stream

KW = dict(panel=("id", "time"), cohort="first_treat", threads=2)


def panel(mode="balanced"):
    rng = np.random.default_rng(420)
    n, T = 360, 5
    ids = np.repeat(np.arange(n), T)
    t = np.tile(np.arange(1, T + 1), n)
    g = np.repeat(np.repeat([0, 3, 5], [100, 120, 140]), T)
    d = pl.DataFrame(
        dict(
            id=ids,
            time=t,
            first_treat=g,
            y=rng.normal(size=n * T) + 0.2 * t + ((g > 0) & (t >= g)),
            w=np.where(ids == 100, 300.0, 1.0),
        )
    )
    if mode == "unbalanced":
        d = d.filter(~((pl.col("id") % 4 == 0) & pl.col("time").is_in([2, 4])))
    return d


@pytest.mark.parametrize("mode", ["balanced", "unbalanced"])
@pytest.mark.parametrize("method", ["reg", "ipw", "dr"])
def test_weight_ess_matches_observed_strata_and_warning_preserves_fit(mode, method):
    data = panel(mode)
    with pytest.warns(LowEffectiveSampleWarning, match="below 30"):
        a = cs_stream("y ~ 1", data, weights="w", panel_mode=mode, est_method=method, **KW)
    b = cs_stream(
        "y ~ 1", data, weights="w", panel_mode=mode, est_method=method, ess_warn_threshold=0, **KW
    )
    try:
        assert_allclose(a.beta, b.beta, atol=0, rtol=0)
        assert_allclose(a.vcov, b.vcov, atol=0, rtol=0)
        table = a.effective_sample_size()
        for row in table.iter_rows(named=True):
            cohort, t = row["cohort"], row["time"]
            base = cohort - 1 if t >= cohort else t - 1
            group = cohort if row["stratum"].startswith("treated") else 0
            date = base if row["stratum"].endswith("_pre") else t
            weights = data.filter((pl.col("first_treat") == group) & (pl.col("time") == date))[
                "w"
            ].to_numpy()
            assert_allclose(
                row["effective_n"], weights.sum() ** 2 / (weights @ weights), atol=1e-10
            )
        assert a.group_time()["min_effective_n"].min() < 30
        assert not a.group_time()["suppressed"].any()
    finally:
        a.cleanup()
        b.cleanup()


@pytest.mark.parametrize("vcov", ["CRV0", "CRV1"])
def test_suppression_is_a_copy_and_does_not_reweight_aggregates(vcov):
    fit = cs_stream(
        "y ~ 1",
        panel(),
        weights="w",
        vcov=vcov,
        ess_warn_threshold=0,
        bootstrap_reps=51,
        bootstrap_seed=2,
        **KW,
    )
    try:
        beta, V = fit.beta.copy(), fit.vcov.copy()
        suppressed = fit.suppress(30)
        mask = np.array([g == 3 for g, t in fit.cells])
        assert np.isnan(suppressed.beta[mask]).all()
        assert np.isnan(suppressed.vcov[mask, :]).all()
        assert_allclose(suppressed.beta[~mask], beta[~mask])
        assert_allclose(suppressed.vcov[np.ix_(~mask, ~mask)], V[np.ix_(~mask, ~mask)])
        assert_allclose(fit.beta, beta)
        assert_allclose(fit.vcov, V)
        again = suppressed.suppress(0)
        assert_allclose(again.beta, suppressed.beta, equal_nan=True)
        assert again.diagnostics["suppression_threshold"] == 30
        assert np.isnan(suppressed.aggregate("simple").beta).all()
        a, b = suppressed.aggregate(), fit.aggregate().suppress(30)
        assert_allclose(a.beta, b.beta, equal_nan=True)
        assert_allclose(a.vcov, b.vcov, equal_nan=True)
        assert "suppressed coefficients" in a.summary_text()
        assert_allclose(
            suppressed.pretrend_test(level="dynamic", horizons=[-3])["statistic"],
            fit.pretrend_test(level="dynamic", horizons=[-3])["statistic"],
        )
        assert_allclose(a.contributions()["weight"], fit.contributions()["weight"])
        assert np.isfinite(a.beta).any() and np.isnan(a.beta).any()
        safe = [name for name, bad in zip(suppressed.coefnames, mask, strict=True) if not bad]
        assert_allclose(
            suppressed.confint(keep=safe, exact_match=True, joint=True, seed=2)
            .drop("Coefficient")
            .to_numpy(),
            fit.confint(keep=safe, exact_match=True, joint=True, seed=2)
            .drop("Coefficient")
            .to_numpy(),
        )
        assert suppressed.wald_test(keep=safe, exact_match=True) == fit.wald_test(
            keep=safe, exact_match=True
        )
        R = np.eye(len(beta))[~mask]
        assert suppressed.wald_test(R=R) == fit.wald_test(R=R)
        with pytest.raises(ValueError, match="suppressed"):
            suppressed.confint(joint=True)
        with pytest.raises(ValueError, match="suppressed"):
            suppressed.wald_test()
        with pytest.raises(ValueError, match="suppressed"):
            suppressed.multiplier_confint()
        assert_allclose(
            suppressed.multiplier_draws(keep=safe, exact_match=True),
            fit.multiplier_draws(keep=safe, exact_match=True),
        )
        assert np.isnan(suppressed.multiplier_draws()[:, mask]).all()
        assert_allclose(suppressed.with_vcov("CRV1").aggregate().beta, b.beta, equal_nan=True)
        assert_allclose(
            suppressed.with_vcov("CRV1").aggregate().vcov,
            fit.with_vcov("CRV1").aggregate().suppress(30).vcov,
            equal_nan=True,
        )
        # Suppression thresholds operate on Kish ESS, hence common weight scale cancels.
        scaled = cs_stream(
            "y ~ 1",
            panel().with_columns(pl.col("w") * 1e100),
            weights="w",
            ess_warn_threshold=None,
            ess_suppress_threshold=30,
            **KW,
        )
        assert_allclose(scaled.beta, suppressed.beta, equal_nan=True, atol=1e-9)
        scaled.cleanup()
    finally:
        fit.cleanup()


def test_unweighted_fast_path_threshold_boundary_and_invalid_options():
    d = panel().filter(pl.col("id") < 220).drop("w")
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        fit = cs_stream("y ~ 1", d, ess_warn_threshold=100, **KW)
    assert not any(isinstance(w.message, LowEffectiveSampleWarning) for w in seen)
    assert fit.effective_sample_size()["effective_n"].min() == 100
    assert np.isnan(fit.suppress(121).beta).all()
    for bad in [-1, True, float("nan"), float("inf"), "30"]:
        with pytest.raises(ValueError, match="ess_warn_threshold"):
            cs_stream("y ~ 1", d, ess_warn_threshold=bad, **KW)
        with pytest.raises(ValueError, match="min_effective_n"):
            fit.suppress(bad)
    with pytest.raises(ValueError, match="CS options"):
        sunab_stream("y ~ 1", d, ess_suppress_threshold=30, **KW)


@pytest.mark.parametrize("control", ["nevertreated", "notyettreated"])
def test_native_unbalanced_fit_time_suppression_equals_post_estimation(control):
    d = panel("unbalanced")
    a = cs_stream(
        "y ~ 1",
        d,
        weights="w",
        panel_mode="unbalanced",
        control_group=control,
        ess_warn_threshold=None,
        **KW,
    )
    b = cs_stream(
        "y ~ 1",
        d,
        weights="w",
        panel_mode="unbalanced",
        control_group=control,
        ess_warn_threshold=None,
        ess_suppress_threshold=30,
        **KW,
    )
    try:
        post = a.suppress(30)
        assert_allclose(post.beta, b.beta, equal_nan=True)
        assert_allclose(post.vcov, b.vcov, equal_nan=True)
        assert np.isfinite(b.beta).any() and np.isnan(b.beta).any()
        assert_allclose(post.aggregate().beta, b.aggregate().beta, equal_nan=True)
    finally:
        a.cleanup()
        b.cleanup()
