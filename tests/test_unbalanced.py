"""Unbalanced panels: external R fixtures, dense oracles and sample rules."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose
from oracles import regression

from benchmarks._estimators import fit_estimator, public
from benchmarks._files import read_csv
from diff_stream import did_bjs, did_callaway_santanna, did_sun_abraham

ROOT = Path(__file__).parent / "data"
REF = ROOT / "unbalanced_reference"
KW = dict(panel=("id", "time"), cohort="first_treat", n_threads=2, incomplete_units="keep")


@pytest.fixture(scope="module")
def unbalanced():
    return read_csv(ROOT / "unbalanced_panel.csv")


def load(name):
    return read_csv(REF / name, null_values="NA")


@pytest.mark.parametrize("formula", ["constant", "numeric"])
@pytest.mark.parametrize("controls", ["nevertreated", "notyettreated"])
@pytest.mark.parametrize("base", ["varying", "universal"])
@pytest.mark.parametrize("method", ["reg", "ipw", "dr"])
def test_r_cs(unbalanced, formula, controls, base, method):
    prefix = f"cs_{formula}_{controls}_{base}_{method}"
    ref = load(prefix + ".csv")
    r = did_callaway_santanna(
        "y ~ 1" if formula == "constant" else "y ~ x + z",
        unbalanced,
        control_group=public("control_group", controls),
        base_period=base,
        adjustment=public("adjustment", method),
        **KW,
    )
    keys = list(zip(ref["g"], ref["t"], strict=True))
    ix = [keys.index(c) for c in r.cells]
    assert_allclose(r.beta, ref["att"].to_numpy()[ix], atol=2e-8)
    V = load(prefix + "_vcov.csv").to_numpy()
    assert_allclose(r.vcov, V[np.ix_(ix, ix)], atol=2e-8)
    for kind in ["dynamic", "simple"]:
        a, z = r.aggregate(public("kind", kind)), load(prefix + "_" + kind + ".csv")
        if kind == "dynamic":
            events = [int(n.split("::")[-1]) for n in a.coefnames]
            z = z.filter(pl.col("event").is_in(events)).sort("event")
        assert_allclose(a.beta, z["att"], atol=2e-8)
        assert_allclose(a.se, z["se"], atol=2e-8)
    assert r.n_obs == len(unbalanced)
    assert r.n_units == unbalanced["id"].n_unique()


def test_r_sunab(unbalanced):
    r = did_sun_abraham("y ~ 1", unbalanced, **KW)
    ref = load("sunab_cells.csv")
    # fixest names interactions by relative time and cohort.
    keys = []
    for name in ref["term"]:
        t, g = name.split(":cohort::")
        g = int(g)
        keys.append((g, int(t.split("::")[-1]) + g))
    ix = [keys.index(c) for c in r.cells]
    assert_allclose(r.beta, ref["att"].to_numpy()[ix], atol=1e-8)
    V = load("sunab_cells_vcov.csv").to_numpy()
    assert_allclose(r.vcov, V[np.ix_(ix, ix)], atol=1e-8)
    a, z = r.aggregate(), load("sunab_dynamic.csv")
    ix = [list(z["term"]).index("time::" + name.split("::")[-1]) for name in a.coefnames]
    assert_allclose(a.beta, z["att"].to_numpy()[ix], atol=1e-8)
    assert_allclose(a.se, z["se"].to_numpy()[ix], atol=1e-8)


@pytest.mark.parametrize("avg", ["cohort_time", "event_time", "cohort", "all"])
@pytest.mark.parametrize("leaveout", [False, True])
def test_r_dense_bjs(unbalanced, avg, leaveout, tmp_path):
    r = did_bjs(
        "y ~ 1", unbalanced, residual_averaging=avg, leave_out=leaveout, workdir=tmp_path, **KW
    )
    try:
        for kind in ["cells", "dynamic", "simple"]:
            prefix = f"bjs_{avg}_{'leaveout' if leaveout else 'plugin'}_{kind}"
            a = r if kind == "cells" else r.aggregate(public("kind", kind))
            ref = load(prefix + ".csv")
            assert a.coefnames == ref["term"].to_list()
            assert_allclose(a.beta, ref["estimate"], atol=1e-9)
            assert_allclose(a.vcov, load(prefix + "_vcov.csv").to_numpy(), atol=1e-9)
        assert list(tmp_path.iterdir())
    finally:
        r.cleanup()
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError, match="cleanup"):
        r.aggregate()
    assert r.tidy()["Estimate"].is_finite().all()


@pytest.mark.parametrize("estimator", ["cs", "sunab", "bjs"])
def test_sample_modes(data, estimator, tmp_path):
    kw = {**KW, "estimator": estimator, "workdir": tmp_path}
    strict = {k: v for k, v in kw.items() if k != "incomplete_units"}
    a = fit_estimator("y ~ 1", data, **strict)
    b = fit_estimator("y ~ 1", data, **kw)
    assert_allclose(a.beta, b.beta, atol=1e-11)
    assert_allclose(a.vcov, b.vcov, atol=1e-11)
    missing = data.filter(~((pl.col("id") % 5 == 0) & (pl.col("time") == 3)))
    with pytest.raises(ValueError, match="balanced"):
        fit_estimator("y ~ 1", missing, **strict)
    complete = missing.filter(pl.col("id") % 5 != 0)
    a = fit_estimator("y ~ 1", complete, **strict)
    b = fit_estimator("y ~ 1", missing, **{**kw, "incomplete_units": "drop"})
    assert_allclose(a.beta, b.beta, atol=1e-11)
    assert_allclose(a.vcov, b.vcov, atol=1e-11)
    assert b.diagnostics["dropped_units"] == 16
    assert b.diagnostics["dropped_rows"] == 64
    assert b.n_obs == len(complete)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("estimator", ["sunab", "bjs"])
def test_dense_observed_regression(data, estimator):
    p = data.filter(~((pl.col("id") % 3 == 0) & (pl.col("time") == 3)))
    r = fit_estimator("y ~ 1", p, estimator=estimator, batch_units=7, **KW)
    try:
        b, V = regression(p, estimator, r.cells)
        assert_allclose(r.beta, b, atol=1e-9)
        assert_allclose(r.vcov, V, atol=1e-9)
    finally:
        r.cleanup()


@pytest.mark.parametrize("estimator", ["cs", "sunab", "bjs"])
def test_partition_batch_shuffle_crv(unbalanced, estimator):
    kw = {**KW, "estimator": estimator}
    a = fit_estimator("y ~ 1", unbalanced, batch_units=9, preprocess_mb=1, **kw)
    b = fit_estimator(
        "y ~ 1", unbalanced.sample(fraction=1, shuffle=True, seed=8), vcov="CRV1", **kw
    )
    try:
        assert_allclose(a.beta, b.beta, atol=1e-9)
        assert_allclose(a.vcov * a.n_units / (a.n_units - 1), b.vcov, atol=1e-9)
        assert_allclose(a.with_vcov("CRV1").aggregate().vcov, b.aggregate().vcov, atol=1e-9)
    finally:
        a.cleanup()
        b.cleanup()


def test_bjs_package_simple(unbalanced):
    r = did_bjs("y ~ 1", unbalanced, **KW)
    try:
        a, ref = r.aggregate("overall"), load("bjs_package_simple.csv")
        assert_allclose(a.beta, ref["estimate"], atol=1e-8)
        assert_allclose(a.se, ref["std.error"], atol=1e-8)
    finally:
        r.cleanup()


@pytest.mark.parametrize("mode", ["complete", "unbalanced"])
def test_invalid_rows_still_rejected(data, mode, tmp_path):
    kw = {**KW, "incomplete_units": public("incomplete_units", mode), "workdir": tmp_path}
    with pytest.raises(ValueError, match="duplicate"):
        did_callaway_santanna("y ~ 1", pl.concat([data, data.head(1)]), **kw)
    with pytest.raises(ValueError, match="missing"):
        did_callaway_santanna(
            "y ~ 1",
            data.with_columns(
                pl.when(pl.col("id") == 0).then(None).otherwise(pl.col("y")).alias("y")
            ),
            **kw,
        )
    changed = data.with_columns(
        pl.when((pl.col("id") == 0) & (pl.col("time") == 2))
        .then(4)
        .otherwise(pl.col("first_treat"))
        .alias("first_treat")
    )
    with pytest.raises(ValueError, match="constant"):
        did_callaway_santanna("y ~ 1", changed, **kw)
    assert not list(tmp_path.iterdir())


def test_bjs_identification_and_failure_cleanup(data, tmp_path):
    g = data.filter(pl.col("first_treat") > 0)["first_treat"][0]
    unit = data.filter(pl.col("first_treat") == g)["id"][0]
    p = data.filter(~((pl.col("id") == unit) & (pl.col("time") < g)))
    with pytest.raises(ValueError, match="untreated observation"):
        did_bjs("y ~ 1", p, workdir=tmp_path, **KW)
    # No untreated rows at the last date leaves its time effect unidentified.
    p = data.filter(~((pl.col("first_treat") == 0) & (pl.col("time") == 5)))
    with pytest.raises(ValueError, match="disconnected"):
        did_bjs("y ~ 1", p, workdir=tmp_path, **KW)
    # A singleton treated cell cannot identify its cluster leave-out variance.
    p = data.filter(
        ~((pl.col("first_treat") == g) & (pl.col("time") == 5) & (pl.col("id") != unit))
    )
    with pytest.raises(ValueError, match="more than one"):
        did_bjs("y ~ 1", p, leave_out=True, workdir=tmp_path, **KW)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("estimator", ["cs", "sunab", "bjs"])
def test_empty_cells_reported(data, estimator):
    g = data.filter(pl.col("first_treat") > 0)["first_treat"][0]
    p = data.filter(~((pl.col("first_treat") == g) & (pl.col("time") == 5)))
    r = fit_estimator("y ~ 1", p, estimator=estimator, **KW)
    try:
        assert (g, 5) not in r.cells
        assert {
            "cohort": g,
            "time": 5,
            "reason": "no observations in a required cohort/date stratum",
        } in r.diagnostics["omitted_cells"]
        assert r.diagnostics["warnings"]
    finally:
        r.cleanup()


def test_sparse_group_batches_cross_reader_boundaries(tmp_path):
    from diff_stream._core.partition import whole_group_batches

    frame = pl.DataFrame({"unit": [0] * 7 + [1] * 2 + [2] * 9 + [3], "value": list(range(19))})
    path = tmp_path / "groups.parquet"
    frame.write_parquet(path, row_group_size=3)
    batches = list(whole_group_batches(path, "unit", 2, read_rows=4))
    assert pl.concat(batches).equals(frame)
    assert sum(b["unit"].n_unique() for b in batches) == 4
    assert all(b["unit"].n_unique() <= 2 for b in batches)


def test_profile_leaves_no_bjs_files(data, tmp_path):
    p = data.filter(~((pl.col("id") % 3 == 0) & (pl.col("time") == 3)))
    profile = did_bjs("y ~ 1", p, profile=True, workdir=tmp_path, **KW)
    assert profile.notes["balanced_sample"] is False
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("estimator", ["cs", "sunab", "bjs"])
def test_recursive_partitions_with_string_units(estimator, tmp_path):
    from diff_stream.simulate import simulate_panel

    p = simulate_panel(5000, 5, seed=82)
    p = p.filter(~((pl.col("id") % 3 == 0) & (pl.col("time") == 3)))
    p = p.with_columns(pl.col("id").cast(pl.String))
    kw = {**KW, "estimator": estimator, "workdir": tmp_path}
    a = fit_estimator("y ~ 1", p, **kw)
    b = fit_estimator(
        "y ~ 1", p.sample(fraction=1, shuffle=True, seed=90), batch_units=31, preprocess_mb=1, **kw
    )
    try:
        assert b.diagnostics["partitions"] > 1
        assert b.diagnostics["max_decoded_leaf_bytes"] <= 250000
        assert_allclose(a.beta, b.beta, atol=1e-9)
        assert_allclose(a.vcov, b.vcov, atol=1e-9)
    finally:
        a.cleanup()
        b.cleanup()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("method", ["reg", "ipw", "dr"])
@pytest.mark.parametrize("trim", [0.4, 0.995, 1])
def test_rc_trimming_and_unit_folding(unbalanced, method, trim):
    # The reference compares cohort 3 to never-treated rows at dates 2 and 4.
    # Keep all dates, but exclude unrelated cohorts whose overlap at .4 differs.
    sample = unbalanced.filter(pl.col("first_treat").is_in([0, 3]))
    r = did_callaway_santanna(
        "y ~ x + z", sample, adjustment=public("adjustment", method), propensity_trim=trim, **KW
    )
    j = r.cells.index((3, 4))
    ref = load(f"trim_{method}_{trim:g}.csv")
    assert_allclose(r.beta[j], ref["att"][0], atol=1e-8)
    assert_allclose(r.se[j], ref["se"][0], atol=1e-8)
    if method != "reg" and trim == 0.4:
        assert any(c["trimmed_controls"] > 0 for c in r.diagnostics["comparisons"])


def test_rc_all_controls_trimmed(unbalanced):
    with pytest.raises(ValueError, match="no contributing observations"):
        did_callaway_santanna("y ~ 1", unbalanced, propensity_trim=0.00001, **KW)


def test_sunab_missing_cohort_reference(data, tmp_path):
    g = data.filter(pl.col("first_treat") > 0)["first_treat"][0]
    p = data.filter(~((pl.col("first_treat") == g) & (pl.col("time") == g - 1)))
    with pytest.raises(ValueError, match="unidentified"):
        did_sun_abraham("y ~ 1", p, workdir=tmp_path, **KW)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("estimator", ["sunab", "bjs"])
def test_observed_singleton_never_unit(data, estimator):
    p = data.filter((pl.col("id") != 0) | (pl.col("time") == 5))
    r = fit_estimator("y ~ 1", p, estimator=estimator, **KW)
    try:
        beta, V = regression(p, estimator, r.cells)
        assert r.n_units == data["id"].n_unique()
        assert_allclose(r.beta, beta, atol=1e-9)
        assert_allclose(r.vcov, V, atol=1e-9)
    finally:
        r.cleanup()
