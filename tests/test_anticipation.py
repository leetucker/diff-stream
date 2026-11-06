"""Anticipation: clean samples, reference periods, actual event labels and scores."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose
from oracles import regression

from benchmarks._files import read_csv
from diff_stream import diff_stream
from diff_stream.simulate import simulate_panel

ROOT = Path(__file__).parent / "data"
REF = ROOT / "anticipation_reference"
KW = dict(panel=("id", "time"), cohort="first_treat", threads=2)


@pytest.fixture(scope="module", params=["balanced", "unbalanced"])
def sample(request):
    name = (
        "anticipation_panel.csv"
        if request.param == "balanced"
        else "anticipation_unbalanced_panel.csv"
    )
    return request.param, read_csv(ROOT / name)


@pytest.mark.parametrize("A", [1, 2])
@pytest.mark.parametrize(
    "design,method", [("constant", "dr"), ("numeric", "reg"), ("numeric", "ipw"), ("numeric", "dr")]
)
@pytest.mark.parametrize("control", ["nevertreated", "notyettreated"])
@pytest.mark.parametrize("base", ["varying", "universal"])
def test_r_cs_anticipation(sample, A, design, method, control, base):
    label, p = sample
    r = diff_stream(
        "y ~ 1" if design == "constant" else "y ~ x + z",
        p,
        anticipation=A,
        est_method=method,
        control_group=control,
        base_period=base,
        panel_mode=label,
        **KW,
    )
    prefix = f"{label}_{A}_{design}_{control}_{base}_{method}"
    ref = read_csv(REF / (prefix + ".csv"), null_values="NA")
    keys = list(zip(ref["g"], ref["t"], strict=True))
    ix = [keys.index(c) for c in r.cells]
    assert_allclose(r.beta, ref["att"].to_numpy()[ix], atol=2e-8)
    V = read_csv(REF / (prefix + "_vcov.csv")).to_numpy()
    assert_allclose(r.vcov, V[np.ix_(ix, ix)], atol=2e-8)
    a = r.aggregate()
    z = read_csv(REF / (prefix + "_dynamic.csv"), null_values="NA")
    levels = [int(n.split("::")[-1]) for n in a.coefnames]
    z = z.filter(pl.col("event").is_in(levels)).sort("event")
    assert_allclose(a.beta, z["att"], atol=2e-8)
    assert_allclose(a.se, z["se"], atol=2e-8)
    if base == "universal":
        for kind in ["simple", "cohort", "calendar"]:
            a = r.aggregate(kind)
            suffix = "group" if kind == "cohort" else kind
            z = read_csv(REF / (prefix + "_" + suffix + ".csv"), null_values="NA")
            assert_allclose(a.beta, z["att"], atol=2e-8)
            assert_allclose(a.se, z["se"], atol=2e-8)
    assert r.diagnostics["reference_event_time"] == -A - 1
    assert "anticipation" in r.group_time()["period_type"]


@pytest.mark.parametrize("A", [1, 2])
def test_fixest_shifted_reference(sample, A):
    label, p = sample
    r = diff_stream("y ~ 1", p, estimator="sunab", anticipation=A, panel_mode=label, **KW)
    ref = read_csv(REF / f"{label}_{A}_sunab.csv")
    keys = []
    for term in ref["term"]:
        event, g = term.split(":cohort::")
        g = int(g)
        keys.append((g, int(event.split("::")[-1]) + g))
    ix = [keys.index(c) for c in r.cells]
    assert_allclose(r.beta, ref["att"].to_numpy()[ix], atol=1e-9)
    V = read_csv(REF / f"{label}_{A}_sunab_vcov.csv").to_numpy()
    assert_allclose(r.vcov, V[np.ix_(ix, ix)], atol=1e-9)
    assert all(t - g != -A - 1 for g, t in r.cells)
    assert any(t - g == -1 for g, t in r.cells)


@pytest.mark.parametrize("A", [1, 2])
@pytest.mark.parametrize("average", ["cohort_time", "event_time", "cohort", "all"])
@pytest.mark.parametrize("leaveout", [False, True])
def test_bjs_smart_anticipation(sample, A, average, leaveout, tmp_path):
    label, p = sample
    r = diff_stream(
        "y ~ 1",
        p,
        estimator="bjs",
        anticipation=A,
        panel_mode=label,
        avgeffectsby=average,
        leaveout=leaveout,
        workdir=tmp_path,
        **KW,
    )
    try:
        for kind in ["cells", "dynamic", "simple"]:
            name = f"bjs_{average}_{'leaveout' if leaveout else 'plugin'}_{kind}"
            folder = REF / f"bjs_{label}_{A}"
            a = r if kind == "cells" else r.aggregate(kind)
            ref = read_csv(folder / (name + ".csv"))
            assert a.coefnames == ref["term"].to_list()
            assert_allclose(a.beta, ref["estimate"], atol=1e-9)
            assert_allclose(a.vcov, read_csv(folder / (name + "_vcov.csv")).to_numpy(), atol=1e-9)
        if average == "cohort_time" and not leaveout:
            for kind in ["dynamic", "simple"]:
                a = r.aggregate(kind)
                ref = read_csv(folder / f"bjs_package_{kind}.csv")
                if kind == "dynamic":
                    ref = ref.with_columns(pl.col("term").cast(pl.Int64)).sort("term")
                assert_allclose(a.beta, ref["estimate"], atol=1e-8)
                assert_allclose(a.se, ref["std.error"], atol=1e-8)
        assert min(t - g for g, t in r.cells) == -A
        assert (r.aggregate("simple").contributions()["event_time"] >= 0).all()
    finally:
        r.cleanup()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("estimator", ["cs", "sunab", "bjs"])
def test_zero_is_identical_and_invalid_values_fail_before_scan(data, estimator, monkeypatch):
    a = diff_stream("y ~ 1", data, estimator=estimator, **KW)
    b = diff_stream("y ~ 1", data, estimator=estimator, anticipation=0, **KW)
    assert_allclose(a.beta, b.beta, atol=1e-12)
    assert_allclose(a.vcov, b.vcov, atol=1e-12)

    def fail(*args, **kwargs):
        raise AssertionError("source executed")

    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", fail)
    for value in [-1, True, np.bool_(True), 1.5, "1", None, 2**64]:
        with pytest.raises(ValueError, match="anticipation"):
            diff_stream("y ~ 1", data, estimator=estimator, anticipation=value, **KW)


@pytest.mark.parametrize("estimator", ["cs", "sunab", "bjs"])
def test_clean_preperiod_required(data, estimator, tmp_path):
    # simulate_panel forces a g=2 cohort. A=1 leaves no clean in-sample base.
    with pytest.raises(ValueError, match="clean pre-period"):
        diff_stream("y ~ 1", data, estimator=estimator, anticipation=1, workdir=tmp_path, **KW)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("estimator", ["cs", "sunab", "bjs"])
def test_future_cohorts_kept_until_anticipation_starts_after_sample(estimator):
    p = simulate_panel(120, 6, seed=34).with_columns(
        pl.when(pl.col("id") < 30)
        .then(0)
        .when(pl.col("id") < 60)
        .then(5)
        .when(pl.col("id") < 90)
        .then(7)
        .otherwise(9)
        .alias("first_treat")
    )
    r = diff_stream("y ~ 1", p, estimator=estimator, anticipation=2, **KW)
    try:
        assert "7" in r.diagnostics["cohort_sizes"] and "9" not in r.diagnostics["cohort_sizes"]
        assert r.diagnostics["cohort_sizes"]["0"] == 60
        assert any(g == 7 for g, t in r.cells)
        assert 7 not in r.aggregate("cohort").contributions()["cohort"]
        assert set(r.group_time().filter(pl.col("cohort") == 7)["period_type"]) <= {
            "pre_treatment",
            "anticipation",
        }
    finally:
        r.cleanup()


@pytest.mark.parametrize("estimator", ["cs", "sunab", "bjs"])
def test_partition_modes_and_post_aggregation(estimator):
    p = read_csv(ROOT / "anticipation_panel.csv")
    a = diff_stream(
        "y ~ 1",
        p,
        estimator=estimator,
        anticipation=np.int64(2),
        batch_units=13,
        preprocess_mb=1,
        **KW,
    )
    missing = p.filter(~((pl.col("id") % 4 == 0) & (pl.col("time") == 7)))
    b = diff_stream(
        "y ~ 1", missing, estimator=estimator, anticipation=2, panel_mode="complete", **KW
    )
    c = diff_stream(
        "y ~ 1", p.filter(pl.col("id") % 4 != 0), estimator=estimator, anticipation=2, **KW
    )
    try:
        assert_allclose(b.beta, c.beta, atol=1e-10)
        assert_allclose(b.vcov, c.vcov, atol=1e-10)
        for kind in ["simple", "cohort", "calendar"]:
            assert (a.aggregate(kind).contributions()["event_time"] >= 0).all()
        w = a.aggregate(balance_window=(-2, 0))
        assert w.coefnames == ["event::-2", "event::-1", "event::0"]
        assert not w.composition()["cohorts_changed"].any()
        assert_allclose(
            a.with_vcov("CRV1").aggregate().vcov, a.aggregate().vcov * a.n_units / (a.n_units - 1)
        )
    finally:
        a.cleanup()
        b.cleanup()
        c.cleanup()


@pytest.mark.parametrize("estimator", ["sunab", "bjs"])
@pytest.mark.parametrize("unbalanced", [False, True])
def test_future_interactions_dense(estimator, unbalanced):
    p = simulate_panel(100, 7, seed=96).with_columns(
        pl.when(pl.col("id") < 40)
        .then(0)
        .when(pl.col("id") < 70)
        .then(5)
        .otherwise(8)
        .alias("first_treat")
    )
    if unbalanced:
        p = p.filter(~((pl.col("id") % 3 == 0) & (pl.col("time") == 4)))
    r = diff_stream("y ~ 1", p, estimator=estimator, anticipation=2, panel_mode="unbalanced", **KW)
    try:
        b, V = regression(p, estimator, r.cells, anticipation=2)
        assert_allclose(r.beta, b, atol=1e-10)
        assert_allclose(r.vcov, V, atol=1e-10)
    finally:
        r.cleanup()


@pytest.mark.parametrize("estimator", ["cs", "sunab", "bjs"])
def test_far_future_labels_do_not_limit_event_window(estimator):
    p = simulate_panel(100, 6, seed=39).with_columns(
        pl.when(pl.col("id") < 40).then(0).otherwise(100).alias("first_treat")
    )
    r = diff_stream("y ~ 1", p, estimator=estimator, anticipation=97, **KW)
    try:
        a = r.aggregate(balance_window=(-97, -94))
        assert a.coefnames == [f"event::{e}" for e in range(-97, -93)]
        assert set(a.contributions()["period_type"]) == {"anticipation"}
        with pytest.raises(ValueError, match="no supported cells"):
            r.aggregate("simple")
    finally:
        r.cleanup()


@pytest.mark.parametrize("estimator", ["cs", "sunab", "bjs"])
def test_anticipation_truth_and_safe_window(estimator):
    p = simulate_panel(120, 8, seed=12).with_columns(
        pl.when(pl.col("id") < 40)
        .then(0)
        .when(pl.col("id") < 80)
        .then(5)
        .otherwise(7)
        .alias("first_treat")
    )
    p = p.with_columns(
        (
            pl.col("id") * 0.3
            + pl.col("time") * 0.2
            + pl.when((pl.col("first_treat") > 0) & (pl.col("time") >= pl.col("first_treat") - 2))
            .then(0.7 + 0.2 * (pl.col("time") - pl.col("first_treat")))
            .otherwise(0)
        ).alias("y")
    )
    r = diff_stream("y ~ 1", p, estimator=estimator, anticipation=2, **KW)
    try:
        for (g, t), beta in zip(r.cells, r.beta, strict=True):
            truth = 0.7 + 0.2 * (t - g) if t >= g - 2 else 0
            assert_allclose(beta, truth, atol=1e-11)
        assert np.max(r.se) < 1e-10
        if estimator != "bjs":
            # Clean-reference -3 is omitted for SA/universal CS, not -1.
            other = diff_stream(
                "y ~ 1", p, estimator=estimator, anticipation=2, base_period="universal", **KW
            )
            a = other.aggregate(balance_window=(-3, 1))
            assert a.coefnames == ["event::-2", "event::-1", "event::0", "event::1"]
            assert not a.composition()["cohorts_changed"].any()
    finally:
        r.cleanup()


def test_unbalanced_clean_observation_and_reference_errors(tmp_path):
    p = read_csv(ROOT / "anticipation_panel.csv")
    no_clean = p.filter(~((pl.col("id") == 1) & (pl.col("time") < 4)))
    with pytest.raises(ValueError, match="untreated observation"):
        diff_stream(
            "y ~ 1",
            no_clean,
            estimator="bjs",
            anticipation=2,
            panel_mode="unbalanced",
            workdir=tmp_path,
            **KW,
        )
    no_reference = p.filter(~((pl.col("first_treat") == 6) & (pl.col("time") == 3)))
    with pytest.raises(ValueError, match="missing cohort reference"):
        diff_stream(
            "y ~ 1",
            no_reference,
            estimator="sunab",
            anticipation=2,
            panel_mode="unbalanced",
            workdir=tmp_path,
            **KW,
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("estimator", ["cs", "sunab", "bjs"])
def test_time_origin_does_not_change_anticipation(estimator):
    p = read_csv(ROOT / "anticipation_unbalanced_panel.csv")
    shifted = p.with_columns(
        (pl.col("time") + 20).alias("time"),
        pl.when(pl.col("first_treat") == 0)
        .then(0)
        .otherwise(pl.col("first_treat") + 20)
        .alias("first_treat"),
    )
    a = diff_stream("y ~ 1", p, estimator=estimator, anticipation=2, panel_mode="unbalanced", **KW)
    b = diff_stream(
        "y ~ 1", shifted, estimator=estimator, anticipation=2, panel_mode="unbalanced", **KW
    )
    try:
        assert_allclose(a.beta, b.beta, atol=1e-10)
        assert_allclose(a.vcov, b.vcov, atol=1e-10)
        assert a.group_time()["event_time"].equals(b.group_time()["event_time"])
    finally:
        a.cleanup()
        b.cleanup()
