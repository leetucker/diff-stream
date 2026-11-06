"""Offline checks against package outputs exported by reference.R."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from benchmarks._files import exists as csv_exists
from benchmarks._files import read_csv
from diff_stream import bjs_stream, cs_stream, sunab_stream

DATA = Path(__file__).parent / "data"
REF = DATA / "r_reference"


def load(name):
    p = REF / name
    if not csv_exists(p):
        pytest.skip("R reference fixture absent; run benchmarks/external/reference.R")
    return read_csv(p, null_values="NA")


def fit(fun, **kwargs):
    return fun(
        "y ~ 1",
        read_csv(DATA / "reference_panel.csv"),
        panel=("id", "time"),
        cohort="first_treat",
        threads=2,
        **kwargs,
    )


@pytest.mark.parametrize("controls", ["nevertreated", "notyettreated"])
@pytest.mark.parametrize("base", ["varying", "universal"])
def test_r_did(controls, base):
    prefix = f"cs_{controls}_{base}"
    ref = load(prefix + ".csv")
    r = fit(cs_stream, control_group=controls, base_period=base)
    lookup = {(row["g"], row["t"]): i for i, row in enumerate(ref.iter_rows(named=True))}
    ix = [lookup[c] for c in r.cells]
    assert_allclose(r.beta, ref["att"].to_numpy()[ix], atol=1e-9)
    assert_allclose(r.se, ref["se"].to_numpy()[ix], atol=1e-9)
    V = load(prefix + "_vcov.csv").to_numpy()
    assert_allclose(r.vcov, V[np.ix_(ix, ix)], atol=1e-9)
    for kind in ["dynamic", "simple"]:
        a = r.aggregate(kind)
        z = load(prefix + "_" + kind + ".csv")
        if kind == "dynamic":
            z = z.filter(pl.col("event").is_in([int(x.split("::")[1]) for x in a.coefnames])).sort(
                "event"
            )
        assert_allclose(a.beta, z["att"], atol=1e-9)
        assert_allclose(a.se, z["se"], atol=1e-9)


def test_r_fixest():
    ref = load("sunab_dynamic.csv")
    r = fit(sunab_stream).aggregate()
    events = [int(s.split("::")[-1]) for s in ref["term"]]
    ix = [events.index(int(s.split("::")[-1])) for s in r.coefnames]
    assert_allclose(r.beta, ref["att"].to_numpy()[ix], atol=1e-9)
    assert_allclose(r.se, ref["se"].to_numpy()[ix], atol=1e-9)


def test_r_bjs_points():
    ref = load("bjs_dynamic.csv")
    r = fit(bjs_stream).aggregate()
    ref = ref.with_columns(pl.col("term").cast(pl.Int64).alias("event")).sort("event")
    assert_allclose(r.beta, ref["estimate"], atol=1e-8)
    assert_allclose(r.se, ref["std.error"], atol=1e-8)


@pytest.mark.parametrize("averaging", ["cohort_time", "event_time", "cohort", "all"])
@pytest.mark.parametrize("leaveout", [False, True])
@pytest.mark.parametrize("kind", ["cells", "dynamic", "simple"])
def test_bjs_dense_r_joint_covariance(averaging, leaveout, kind):
    prefix = f"bjs_{averaging}_{'leaveout' if leaveout else 'plugin'}_{kind}"
    ref = load(prefix + ".csv")
    r = fit(bjs_stream, avgeffectsby=averaging, leaveout=leaveout)
    if kind != "cells":
        r = r.aggregate(kind)
    assert r.coefnames == ref["term"].to_list()
    assert_allclose(r.beta, ref["estimate"], atol=1e-10)
    assert_allclose(r.se, ref["se"], atol=1e-10)
    assert_allclose(r.vcov, load(prefix + "_vcov.csv").to_numpy(), atol=1e-10)


def test_bjs_r_package_simple_se():
    ref = load("bjs_package_simple.csv")
    r = fit(bjs_stream).aggregate("simple")
    assert_allclose(r.beta, ref["estimate"], atol=1e-8)
    assert_allclose(r.se, ref["std.error"], atol=1e-8)


def test_stata_bjs_bundle_when_available():
    import os

    directory = Path(
        os.environ.get("DIFF_STATA_BJS", DATA.parents[1] / "benchmarks/results/stata_bjs")
    )
    if not csv_exists(directory / "bjs_cohort_time_plugin_dynamic.csv"):
        pytest.skip(
            "Run benchmarks/external/validate_bjs.do externally to supply authors' Stata results"
        )
    for averaging in ["cohort_time", "event_time", "cohort", "all"]:
        for leaveout in [False, True]:
            r = fit(bjs_stream, avgeffectsby=averaging, leaveout=leaveout)
            for kind in ["dynamic", "simple"]:
                prefix = f"bjs_{averaging}_{'leaveout' if leaveout else 'plugin'}_{kind}"
                ref = read_csv(directory / (prefix + ".csv"))
                labels = [
                    "ATT" if t == "tau" else "event::" + t.removeprefix("tau") for t in ref["term"]
                ]
                a = r.aggregate(kind)
                ix = [labels.index(c) for c in a.coefnames]
                assert_allclose(a.beta, ref["estimate"].to_numpy()[ix], atol=1e-7)
                V = read_csv(directory / (prefix + "_vcov.csv")).to_numpy()
                assert_allclose(a.vcov, V[np.ix_(ix, ix)], atol=1e-7)
