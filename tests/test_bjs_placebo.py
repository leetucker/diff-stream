"""BJS clean-only lead regressions: full covariance, timing and separate inference."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose
from scipy.stats import chi2

from benchmarks._files import read_csv
from diff_stream import BJSPlaceboResult, bjs_stream, cs_stream
from diff_stream.simulate import simulate_panel
from diff_stream.store import PanelStore

ROOT = Path(__file__).parent / "data"
REF = ROOT / "bjs_placebo_reference"
KW = dict(panel=("id", "time"), cohort="first_treat", threads=2)


def sample(mode):
    d = read_csv(ROOT / "anticipation_panel.csv").with_columns((0.2 + pl.col("id") % 9).alias("w"))
    if mode in ("unbalanced", "complete"):
        d = d.filter(~((pl.col("id") % 4 == 0) & pl.col("time").is_in([3, 7])))
    if mode == "future":
        d = d.with_columns(
            pl.when((pl.col("first_treat") == 0) & (pl.col("id") % 2 == 0))
            .then(9)
            .otherwise(pl.col("first_treat"))
            .alias("first_treat")
        )
    return d


@pytest.mark.parametrize("mode", ["balanced", "unbalanced", "complete", "future"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("A", [0, 2])
@pytest.mark.parametrize("L", [1, 2, 3])
def test_r_dense_fixest_and_package(mode, weighted, A, L, tmp_path):
    d = sample(mode)
    f = bjs_stream(
        "y ~ 1",
        d,
        panel_mode="balanced" if mode == "future" else mode,
        weights="w" if weighted else None,
        anticipation=A,
        pretrends=L,
        workdir=tmp_path,
        **KW,
    )
    try:
        p = f.placebos()
        assert isinstance(p, BJSPlaceboResult)
        prefix = f"{mode}_{int(weighted)}_{A}_{L}"
        ref = read_csv(REF / f"{prefix}.csv")
        assert p.coefnames == [f"event::{h}" for h in ref["event"]]
        assert_allclose(p.beta, ref["estimate"], atol=2e-9)
        assert_allclose(p.vcov, read_csv(REF / f"{prefix}_vcov.csv").to_numpy(), atol=2e-9)
        pkg = read_csv(REF / f"{prefix}_package.csv")
        assert_allclose(p.beta, pkg["estimate"], atol=1e-8)
        assert_allclose(p.se, pkg["std.error"], atol=1e-8)
        reference = read_csv(REF / "tests.csv").filter(pl.col("case") == prefix).row(0, named=True)
        test = f.pretrend_test()
        assert_allclose(test["statistic"], reference["statistic"], atol=2e-8)
        assert_allclose(test["pvalue"], reference["pvalue"], atol=2e-9)
        assert test["df1"] == L
        assert test["level"] == "dynamic"
        assert test["reference_event_time"] is None
        assert test["fitted_horizons"] == list(range(-A - L, -A))
        assert p.n_obs == reference["clean_rows"]
        assert p.n_units == reference["n_units"]
        assert p.diagnostics["reference_observations"] == reference["reference_rows"]
        assert all(t["time"] - t["cohort"] < -A for t in test["cells"])
        support = p.support()
        assert (support["event_time"] < -A).all()
        if mode == "future" and A == 0:
            assert support.filter(pl.col("cohort") == 9).height > 0
            assert all(g != 9 for g, t in f.cells)
        assert f.aggregate("simple").pretrend_test() == test
        f.cleanup()
        assert f.placebos().pretrend_test() == test
        assert p.confint(joint=True, reps=101, seed=3).height == L
    finally:
        f.cleanup()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("mode", ["balanced", "unbalanced"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("average", ["cohort_time", "event_time", "cohort", "all"])
@pytest.mark.parametrize("leaveout", [False, True])
def test_post_effects_unchanged_and_placebos_ignore_smart_settings(
    mode, weighted, average, leaveout, tmp_path
):
    d = sample(mode)
    kw = dict(
        panel_mode=mode,
        weights="w" if weighted else None,
        anticipation=2,
        avgeffectsby=average,
        leaveout=leaveout,
        workdir=tmp_path,
        **KW,
    )
    plain = bjs_stream("y ~ 1", d, **kw)
    fitted = bjs_stream("y ~ 1", d, pretrends=2, **kw)
    try:
        assert fitted.cells == plain.cells
        assert_allclose(fitted.beta, plain.beta, atol=1e-10)
        assert_allclose(fitted.vcov, plain.vcov, atol=1e-10)
        for kind in ("dynamic", "simple", "cohort", "calendar"):
            a, b = fitted.aggregate(kind), plain.aggregate(kind)
            assert_allclose(a.beta, b.beta, atol=1e-10)
            assert_allclose(a.vcov, b.vcov, atol=1e-10)
        ref = read_csv(REF / f"{mode}_{int(weighted)}_2_2.csv")
        assert_allclose(fitted.placebos().beta, ref["estimate"], atol=1e-9)
        assert_allclose(fitted.placebos().se, ref["se"], atol=1e-9)
    finally:
        plain.cleanup()
        fitted.cleanup()
    assert not list(tmp_path.iterdir())


def test_horizon_selection_crv_scale_batches_and_support(tmp_path):
    d = sample("unbalanced")
    kw = dict(panel_mode="unbalanced", weights="w", anticipation=2, pretrends=3, **KW)
    a = bjs_stream("y ~ 1", d, workdir=tmp_path, **kw)
    b = bjs_stream(
        "y ~ 1",
        d.with_columns(pl.col("w") * 1e120).sample(fraction=1, shuffle=True, seed=19),
        batch_units=13,
        preprocess_mb=1,
        vcov="CRV1",
        workdir=tmp_path,
        **kw,
    )
    try:
        assert_allclose(a.placebos().beta, b.placebos().beta, atol=1e-10)
        assert_allclose(a.placebos().vcov * 400 / 399, b.placebos().vcov, atol=1e-10)
        assert_allclose(a.with_vcov("CRV1").placebos().vcov, b.placebos().vcov, atol=1e-10)
        assert_allclose(
            a.aggregate().with_vcov("CRV1").placebos().vcov, b.placebos().vcov, atol=1e-10
        )
        p = a.placebos()
        test = p.pretrend_test(horizons=[-5, -3])
        ix = [0, 2]
        bb = p.beta[ix]
        V = p.vcov[np.ix_(ix, ix)]
        W = bb @ np.linalg.solve(V, bb)
        assert_allclose(test["statistic"], W)
        assert_allclose(test["pvalue"], chi2.sf(W, 2))
        assert test["horizons"] == [-5, -3]
        assert test["fitted_horizons"] == [-5, -4, -3]
        support = p.support()
        for row in support.iter_rows(named=True):
            rows = d.filter(
                (pl.col("first_treat") == row["cohort"]) & (pl.col("time") == row["time"])
            )
            assert row["observed_units"] == len(rows)
            assert_allclose(row["weight_total"], rows["w"].sum(), rtol=1e-12)
        for bad in [[-2], [-1], [0], [-99], [], [True]]:
            with pytest.raises(ValueError, match="clean|not estimated"):
                p.pretrend_test(horizons=bad)
        with pytest.raises(ValueError, match="dynamic"):
            a.pretrend_test(level="cell")
    finally:
        a.cleanup()
        b.cleanup()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("bad", [True, -1, 1.5, "2", None, 2**63])
def test_invalid_pretrends_before_source(bad, monkeypatch):
    def no_scan(*args, **kwargs):
        raise AssertionError("executed source")

    monkeypatch.setattr(PanelStore, "__init__", no_scan)
    with pytest.raises(ValueError, match="pretrends"):
        bjs_stream("y ~ 1", sample("balanced"), pretrends=bad, **KW)


def test_not_enabled_or_wrong_estimator():
    f = bjs_stream("y ~ 1", sample("balanced"), **KW)
    with pytest.raises(ValueError, match="pretrends"):
        f.placebos()
    with pytest.raises(ValueError, match="pretrends"):
        f.pretrend_test()
    with pytest.raises(ValueError, match="only for BJS"):
        cs_stream("y ~ 1", sample("balanced"), pretrends=2, **KW)
    with pytest.raises(ValueError, match="only for BJS"):
        cs_stream("y ~ 1", sample("balanced"), **KW).placebos()


def test_identification_errors_and_guard_cleanup(tmp_path, monkeypatch):
    d = simulate_panel(80, 5, seed=13).with_columns(
        pl.when(pl.col("first_treat") != 0).then(3).otherwise(0).alias("first_treat")
    )
    with pytest.raises(ValueError, match="placebo design is unidentified"):
        bjs_stream("y ~ 1", d, pretrends=2, workdir=tmp_path, **KW)
    with pytest.raises(ValueError, match="not observed"):
        bjs_stream("y ~ 1", sample("balanced"), pretrends=20, workdir=tmp_path, **KW)
    with pytest.raises(MemoryError, match="numerical buffers"):
        bjs_stream(
            "y ~ 1",
            sample("balanced"),
            pretrends=10**6,
            memory_gb=1,
            preprocess_mb=1,
            workdir=tmp_path,
            **KW,
        )
    d = sample("unbalanced").filter(~((pl.col("id") == 1) & (pl.col("time") < 6)))
    with pytest.raises(ValueError, match="at least one untreated"):
        bjs_stream("y ~ 1", d, pretrends=2, panel_mode="unbalanced", workdir=tmp_path, **KW)
    assert not list(tmp_path.iterdir())
    from diff_stream import api

    def no_fit(*args, **kwargs):
        raise AssertionError("fit while profiling")

    monkeypatch.setattr(api, "fit_placebos", no_fit)
    profile = bjs_stream("y ~ 1", sample("balanced"), pretrends=3, profile=True, **KW)
    assert profile.notes["pretrends"] == 3


def test_deterministic_leads_and_effect_sample_independence():
    # Construct clean y = unit + time + known lead indicators. A separate clean
    # regression recovers them; averaging the ordinary first-stage residuals
    # would not recover these coefficients.
    d = sample("balanced")
    event = pl.col("time") - pl.col("first_treat")
    d = d.with_columns(
        (
            0.2 * pl.col("id")
            + 0.1 * pl.col("time")
            + pl.when((pl.col("first_treat") > 0) & (event == -3)).then(0.7).otherwise(0)
            + pl.when((pl.col("first_treat") > 0) & (event == -4)).then(-0.4).otherwise(0)
        ).alias("y")
    )
    a = bjs_stream("y ~ 1", d, pretrends=2, anticipation=2, **KW)
    d2 = d.with_columns(
        (
            pl.col("y")
            + pl.when((pl.col("first_treat") > 0) & (event >= -2))
            .then(1000 + pl.col("id") * 3)
            .otherwise(0)
        ).alias("y")
    )
    b = bjs_stream("y ~ 1", d2, pretrends=2, anticipation=2, **KW)
    assert_allclose(a.placebos().beta, [-0.4, 0.7], atol=1e-11)
    assert_allclose(a.placebos().beta, b.placebos().beta, atol=1e-11)
    assert_allclose(a.placebos().vcov, b.placebos().vcov, atol=1e-20)
    assert not np.allclose(a.beta, b.beta)


def test_far_future_event_labels_and_shifted_origin():
    d = simulate_panel(80, 6, seed=13)
    d = d.with_columns(
        pl.when(pl.col("first_treat") > 0).then(100).otherwise(0).alias("first_treat")
    )
    d = d.with_columns(
        (
            0.2 * pl.col("id")
            + 0.1 * pl.col("time")
            + pl.when((pl.col("first_treat") > 0) & (pl.col("time") == 2)).then(0.9).otherwise(0)
        ).alias("y")
    )
    a = bjs_stream("y ~ 1", d, pretrends=1, anticipation=97, **KW)
    assert a.placebos().coefnames == ["event::-98"]
    assert_allclose(a.placebos().beta, [0.9], atol=1e-11)
    shifted = d.with_columns(
        pl.col("time") + 20,
        pl.when(pl.col("first_treat") > 0)
        .then(pl.col("first_treat") + 20)
        .otherwise(0)
        .alias("first_treat"),
    )
    b = bjs_stream("y ~ 1", shifted, pretrends=1, anticipation=97, **KW)
    assert_allclose(a.placebos().beta, b.placebos().beta, atol=1e-11)
    assert_allclose(a.placebos().vcov, b.placebos().vcov, atol=1e-20)
