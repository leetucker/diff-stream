import json

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose
from oracles import cs, regression

from diff_stream import bjs_stream, cs_stream, diff_stream, sunab_stream
from diff_stream.simulate import simulate_panel


@pytest.mark.parametrize("method", ["sunab", "bjs"])
def test_dense_regression_and_covariance(data, method):
    r = diff_stream(
        "y ~ 1",
        data,
        panel=("id", "time"),
        cohort="first_treat",
        estimator=method,
        threads=2,
        batch_units=17,
    )
    b, V = regression(data, method, r.cells)
    assert_allclose(r.beta, b, atol=1e-11)
    assert_allclose(r.vcov, V, atol=1e-11)


@pytest.mark.parametrize("controls", ["nevertreated", "notyettreated"])
@pytest.mark.parametrize("base", ["varying", "universal"])
def test_cs_independent_differences(data, controls, base):
    r = cs_stream(
        "y ~ 1",
        data,
        panel=("id", "time"),
        cohort="first_treat",
        control_group=controls,
        base_period=base,
        threads=2,
    )
    b, scores = cs(data, r.cells, controls, base)
    assert_allclose(r.beta, b, atol=1e-12)
    assert_allclose(r.vcov, scores.T @ scores, atol=1e-12)
    # Independently assemble aggregation IF, including estimated shares.
    gvec = data["first_treat"].to_numpy().reshape(-1, 5)[:, 0]
    selected = np.array([t >= g for g, t in r.cells])
    weights = r.counts * selected
    denom = weights.sum()
    A = weights / denom
    extra = np.zeros(len(gvec))
    for j, (g, _t) in enumerate(r.cells):
        if selected[j]:
            extra += (gvec == g) * (b[j] - A @ b) / denom
    aggregate_scores = scores @ A + extra
    a = r.aggregate("simple")
    assert_allclose(a.beta, A @ b)
    assert_allclose(a.vcov[0, 0], aggregate_scores @ aggregate_scores)


@pytest.mark.parametrize("fit", [bjs_stream, cs_stream, sunab_stream])
def test_exact_heterogeneous_effects(fit):
    df = simulate_panel(100, 6, noise=0)
    r = fit("y ~ 1", df, panel=("id", "time"), cohort="first_treat", threads=2)
    for (g, t), b in zip(r.cells, r.beta, strict=True):
        truth = 0.5 + 0.15 * g + 0.25 * (t - g) if t >= g else 0
        assert_allclose(b, truth, atol=1e-12)
    assert np.max(r.se) < 1e-12


@pytest.mark.parametrize("method", ["bjs", "sunab", "cs"])
def test_sources_batches_threads_and_cleanup(data, tmp_path, method):
    data[:100].write_parquet(tmp_path / "a.parquet", row_group_size=7)
    data[100:].write_parquet(tmp_path / "b.parquet", row_group_size=11)
    kwargs = dict(panel=("id", "time"), cohort="first_treat", estimator=method)
    a = diff_stream("y ~ 1", data, threads=1, batch_units=1, **kwargs)
    shuffled = data.sample(fraction=1, shuffle=True, seed=3).with_columns(
        pl.col("id").cast(pl.String)
    )
    b = diff_stream("y ~ 1", shuffled.lazy(), threads=2, batch_units=13, **kwargs)
    c = diff_stream("y ~ 1", str(tmp_path / "*.parquet"), workdir=tmp_path / "work", **kwargs)
    for other in [b, c]:
        assert_allclose(a.beta, other.beta, atol=1e-12)
        assert_allclose(a.vcov, other.vcov, atol=1e-12)
    assert list((tmp_path / "work").iterdir()) == []


def test_reporting_aggregation_and_profile(data):
    r = sunab_stream("y ~ 1", data, panel=("id", "time"), cohort="first_treat", threads=2)
    assert r.group_time().height == len(r.cells)
    a = r.aggregate(horizons=[0, 1])
    assert a.coefnames == ["event::0", "event::1"]
    assert a.confint().height == 2
    assert a.confint(joint=True, reps=100, seed=1).height == 2
    assert a.wald_test()["distribution"] == "chi2"
    assert json.loads(r.summary_json(coefficients=True))["estimator"] == "sunab"
    p = cs_stream("y ~ 1", data, panel=("id", "time"), cohort="first_treat", profile=True)
    assert p.notes["n_units"] == 80
    assert p.disk_gb[0] > 0
    assert p.recommend().ncpus > 0


def test_crv1(data):
    kw = dict(panel=("id", "time"), cohort="first_treat", threads=2)
    a, b = cs_stream("y ~ 1", data, **kw), cs_stream("y ~ 1", data, vcov="CRV1", **kw)
    assert_allclose(b.vcov, a.vcov * 80 / 79)
    assert_allclose(b.aggregate("simple").vcov, a.aggregate("simple").vcov * 80 / 79)
    assert_allclose(a.with_vcov("CRV1").aggregate("simple").vcov, b.aggregate("simple").vcov)


def test_base_period_support(data):
    kw = dict(panel=("id", "time"), cohort="first_treat", threads=2)
    varying = cs_stream("y ~ 1", data, **kw)
    universal = cs_stream("y ~ 1", data, base_period="universal", **kw)
    assert (3, 2) in varying.cells
    assert (3, 2) not in universal.cells
    assert (3, 1) in universal.cells
    assert all(t > 1 for g, t in varying.cells)


@pytest.mark.parametrize(
    "transform,match",
    [
        (lambda d: d.head(d.height - 1), "balanced"),
        (lambda d: pl.concat([d, d.head(1)]), "balanced"),
        (
            lambda d: d.with_columns(
                pl.when(pl.col("time") == 2)
                .then(0)
                .otherwise(pl.col("first_treat"))
                .alias("first_treat")
            ),
            "constant",
        ),
        (lambda d: d.with_columns(pl.lit(1).alias("first_treat")), "always-treated"),
        (lambda d: d.with_columns(pl.lit(0).alias("first_treat")), "never-treated"),
        (lambda d: d.with_columns(pl.lit(float("nan")).alias("y")), "non-finite"),
        (lambda d: d.with_columns(pl.lit(None).cast(pl.Float64).alias("y")), "missing"),
        (lambda d: d.with_columns((pl.col("time") * 2).alias("time")), "consecutive"),
        (lambda d: d.with_columns(pl.col("time").cast(pl.Float64)), "integers"),
    ],
)
def test_panel_errors(data, transform, match):
    with pytest.raises(ValueError, match=match):
        cs_stream("y ~ 1", transform(data), panel=("id", "time"), cohort="first_treat")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"estimator": "bad"},
        {"control_group": "bad"},
        {"base_period": "bad"},
        {"vcov": "iid"},
        {"batch_units": 0},
        {"threads": 0},
        {"estimator": "sunab", "control_group": "notyettreated"},
    ],
)
def test_option_errors(data, kwargs):
    with pytest.raises(ValueError):
        diff_stream("y ~ 1", data, panel=("id", "time"), cohort="first_treat", **kwargs)


def test_limits_and_unsupported_formula(data):
    kw = dict(panel=("id", "time"), cohort="first_treat")
    with pytest.raises(MemoryError):
        cs_stream("y ~ 1", data, memory_gb=1e-5, **kw)
    with pytest.raises(ValueError, match="unknown variables"):
        cs_stream("y ~ x", data, **kw)
    with pytest.raises(ValueError, match="unknown variables"):
        bjs_stream("y ~ x", data, **kw)


def test_explicit_streaming(data, monkeypatch):
    original = pl.LazyFrame.sink_parquet
    calls = []

    def sink(self, *args, **kwargs):
        calls.append(kwargs.get("engine"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", sink)
    cs_stream("y ~ 1", data, panel=("id", "time"), cohort="first_treat", threads=2)
    assert calls and set(calls) == {"streaming"}


def test_run_cleanup_on_exception(data, monkeypatch, tmp_path):
    import diff_stream.api as api

    def fail(*args):
        raise RuntimeError("injected numerical failure")

    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("unrelated file")
    monkeypatch.setattr(api, "fit_means", fail)
    with pytest.raises(RuntimeError, match="injected"):
        cs_stream("y ~ 1", data, panel=("id", "time"), cohort="first_treat", workdir=tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep.txt"]


def test_future_cohorts_and_original_cohort_validation(data):
    kw = dict(panel=("id", "time"), cohort="first_treat", threads=2)
    future = data.with_columns(
        pl.when(pl.col("first_treat") == 0)
        .then(99)
        .otherwise(pl.col("first_treat"))
        .alias("first_treat")
    )
    a, b = cs_stream("y ~ 1", data, **kw), cs_stream("y ~ 1", future, **kw)
    assert_allclose(a.beta, b.beta)
    assert_allclose(a.vcov, b.vcov)
    changing = future.with_columns(
        pl.when((pl.col("first_treat") == 99) & (pl.col("time") == 1))
        .then(100)
        .otherwise(pl.col("first_treat"))
        .alias("first_treat")
    )
    with pytest.raises(ValueError, match="constant"):
        cs_stream("y ~ 1", changing, **kw)


def test_aggregation_errors(data):
    r = bjs_stream("y ~ 1", data, panel=("id", "time"), cohort="first_treat", threads=2)
    with pytest.raises(ValueError, match="no supported cells"):
        r.aggregate(horizons=[-100])
    with pytest.raises(ValueError, match="kind"):
        r.aggregate("unknown")
    with pytest.raises(ValueError, match="original"):
        r.aggregate().aggregate()
