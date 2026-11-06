"""Sampling weights: external references, joint scores and sample conventions."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from benchmarks._files import read_csv
from diff_stream import diff_stream

ROOT = Path(__file__).parent / "data"
REF = ROOT / "weights_reference"
KW = dict(panel=("id", "time"), cohort="first_treat", threads=2, weights="w")


@pytest.fixture(scope="module", params=["balanced", "unbalanced"])
def sample(request):
    name = (
        "anticipation_panel.csv"
        if request.param == "balanced"
        else "anticipation_unbalanced_panel.csv"
    )
    return request.param, read_csv(ROOT / name).with_columns((0.2 + pl.col("id") % 9).alias("w"))


@pytest.mark.parametrize("A", [0, 2])
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
    if A:
        assert "anticipation" in r.group_time()["period_type"]


@pytest.mark.parametrize("A", [0, 2])
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
    if A:
        assert any(t - g == -1 for g, t in r.cells)


@pytest.mark.parametrize("A", [0, 2])
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
            for kind in ["dynamic", "simple"] if A else ["simple"]:
                a = r.aggregate(kind)
                ref = read_csv(folder / f"bjs_package_{kind}.csv")
                if kind == "dynamic":
                    ref = ref.with_columns(pl.col("term").cast(pl.Int64)).sort("term")
                assert_allclose(a.beta, ref["estimate"], atol=1e-8)
                # Package weighted variance is not the WLS influence oracle.
                # The independent dense R covariance above checks inference.
        assert min(t - g for g, t in r.cells) == -A
        assert (r.aggregate("simple").contributions()["event_time"] >= 0).all()
    finally:
        r.cleanup()
    assert not list(tmp_path.iterdir())


CASES = [("cs", "y ~ 1"), ("cs", "y ~ x + z"), ("sunab", "y ~ 1"), ("bjs", "y ~ 1")]


@pytest.mark.parametrize("estimator,fml", CASES)
def test_equal_weights_and_complete_sample(sample, estimator, fml, tmp_path):
    label, d = sample
    d = d.with_columns(pl.lit(7.0).alias("w"))
    kw = dict(
        panel=("id", "time"),
        cohort="first_treat",
        estimator=estimator,
        panel_mode=label,
        anticipation=2,
        threads=2,
        workdir=tmp_path,
    )
    a = diff_stream(fml, d, **kw)
    b = diff_stream(fml, d, weights="w", **kw)
    try:
        assert_allclose(a.beta, b.beta, atol=1e-10)
        assert_allclose(a.vcov, b.vcov, atol=1e-10)
        for kind in ["dynamic", "cohort", "calendar", "simple"]:
            aa, bb = a.aggregate(kind), b.aggregate(kind)
            assert_allclose(aa.beta, bb.beta, atol=1e-10)
            assert_allclose(aa.vcov, bb.vcov, atol=1e-10)
        assert_allclose(b.group_time()["weight_total"], 7 * b.group_time()["weight_units"])
    finally:
        a.cleanup()
        b.cleanup()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("estimator,fml", CASES)
def test_scale_batch_partition_crv_and_counts(sample, estimator, fml, tmp_path):
    label, d = sample
    kw = dict(estimator=estimator, panel_mode=label, anticipation=2, workdir=tmp_path, **KW)
    a = diff_stream(fml, d, **kw)
    try:
        for scale in [1e-200, 1e200]:
            b = diff_stream(
                fml,
                d.with_columns(pl.col("w") * scale).sample(fraction=1, shuffle=True, seed=28),
                batch_units=13,
                preprocess_mb=1,
                vcov="CRV1",
                **kw,
            )
            try:
                factor = a.n_units / (a.n_units - 1)
                assert_allclose(a.beta, b.beta, atol=1e-9)
                assert_allclose(a.vcov * factor, b.vcov, atol=1e-9)
                assert_allclose(a.with_vcov("CRV1").aggregate().vcov, b.aggregate().vcov, atol=1e-9)
                assert_allclose(
                    a.group_time()["weight_total"] * scale,
                    b.group_time()["weight_total"],
                    rtol=1e-12,
                )
            finally:
                b.cleanup()
        table = a.group_time()
        for row in table.iter_rows(named=True):
            rows = d.filter(pl.col("first_treat") == row["cohort"])
            if estimator == "cs":
                rows = rows.unique("id")
            else:
                rows = rows.filter(pl.col("time") == row["time"])
            assert row["weight_units"] == len(rows)
            assert_allclose(row["weight_total"], rows["w"].sum(), rtol=1e-12)
        for kind in ["simple", "dynamic"]:
            c = a.contributions(kind)
            assert_allclose(c["weighted_estimate"].sum(), a.aggregate(kind).beta.sum())
        assert a.diagnostics["sampling_weights"] == "w"
        assert a.aggregate().diagnostics["weight_basis"].endswith("sampling-weight totals")
    finally:
        a.cleanup()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("estimator,fml", CASES)
def test_complete_panel_weights(estimator, fml, tmp_path):
    d = read_csv(ROOT / "anticipation_unbalanced_panel.csv").with_columns(
        (0.2 + pl.col("id") % 9).alias("w")
    )
    complete = d.filter(pl.col("time").count().over("id") == 8)
    a = diff_stream(fml, d, estimator=estimator, panel_mode="complete", workdir=tmp_path, **KW)
    b = diff_stream(fml, complete, estimator=estimator, workdir=tmp_path, **KW)
    try:
        assert_allclose(a.beta, b.beta, atol=1e-10)
        assert_allclose(a.aggregate().vcov, b.aggregate().vcov, atol=1e-10)
        assert a.diagnostics["dropped_units"] > 0
        assert_allclose(a.diagnostics["total_unit_weight"], complete.unique("id")["w"].sum())
    finally:
        a.cleanup()
        b.cleanup()


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("-inf"), float("nan"), None])
def test_invalid_values_clean_workspace(bad, tmp_path):
    d = read_csv(ROOT / "anticipation_panel.csv").with_columns(
        pl.lit(bad, dtype=pl.Float64).alias("w")
    )
    with pytest.raises(ValueError, match="positive|missing|finite"):
        diff_stream("y ~ 1", d, workdir=tmp_path, **KW)
    assert not list(tmp_path.iterdir())


def test_varying_weights_and_invalid_schema(tmp_path):
    d = read_csv(ROOT / "anticipation_panel.csv")
    varying = d.with_columns(pl.col("time").cast(pl.Float64).alias("w"))
    with pytest.raises(ValueError, match="constant within unit"):
        diff_stream("y ~ 1", varying, workdir=tmp_path, **KW)
    for expression in [pl.lit("1").alias("w"), pl.lit(True).alias("w")]:
        with pytest.raises(ValueError, match="numeric"):
            diff_stream("y ~ 1", d.with_columns(expression), **KW)
    for bad in [True, 12, [], "", "time"]:
        with pytest.raises(ValueError, match="column name|distinct"):
            diff_stream("y ~ 1", d, panel=("id", "time"), cohort="first_treat", weights=bad)
    assert not list(tmp_path.iterdir())
