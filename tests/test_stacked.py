"""Stacked DiD against an explicit stack and its weighted regression."""

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from diff_stream import StackedResult, did_stacked
from diff_stream.simulate import simulate_panel

KW = dict(panel=("id", "time"), cohort="first_treat", n_threads=2)


def stack(d, lo, hi, anticipation=0, weight=None):
    """Wing, Freedman and Hollingsworth's stacked data with corrective weights."""
    first, last = d["time"].min(), d["time"].max()
    pieces = []
    for a in sorted(c for c in d["first_treat"].unique() if c > 0):
        if a + lo < first or a + hi > last:
            continue
        clean = (pl.col("first_treat") == 0) | (pl.col("first_treat") - anticipation > a + hi)
        rows = d.filter(
            ((pl.col("first_treat") == a) | clean) & pl.col("time").is_between(a + lo, a + hi)
        ).with_columns(
            pl.lit(a).alias("sub"),
            (pl.col("first_treat") == a).cast(pl.Int8).alias("D"),
            (pl.col("time") - a).alias("e"),
        )
        if rows.filter(pl.col("D") == 0).height:
            pieces.append(rows)
    s = pl.concat(pieces)
    w = pl.col(weight) if weight else pl.lit(1.0)
    units = (
        s.group_by("sub", "D", "id")
        .agg(w.first().alias("uw"))
        .group_by("sub", "D")
        .agg(pl.col("uw").sum().alias("n"))
    )
    total = units.group_by("D").agg(pl.col("n").sum().alias("N"))
    q = units.join(total, on="D").with_columns((pl.col("n") / pl.col("N")).alias("share"))
    treated = q.filter(pl.col("D") == 1).select("sub", pl.col("share").alias("sd"))
    control = q.filter(pl.col("D") == 0).select("sub", pl.col("share").alias("sc"))
    s = s.join(treated, on="sub").join(control, on="sub")
    return s.with_columns(
        (pl.when(pl.col("D") == 1).then(1.0).otherwise(pl.col("sd") / pl.col("sc")) * w).alias("Q")
    )


def regression(s, lo, hi, ref):
    """WLS of y on D, event-time dummies and their interactions; CRV0 by unit."""
    events = [e for e in range(lo, hi + 1) if e != ref]
    e, D = s["e"].to_numpy(), s["D"].to_numpy()
    X = np.column_stack(
        [np.ones(len(e)), D]
        + [(e == h).astype(float) for h in events]
        + [((e == h) & (D == 1)).astype(float) for h in events]
    )
    y, q, ids = s["y"].to_numpy(), s["Q"].to_numpy(), s["id"].to_numpy()
    bread = np.linalg.inv(X.T @ (q[:, None] * X))
    b = bread @ X.T @ (q * y)
    u = q * (y - X @ b)
    scores = np.zeros((len(np.unique(ids)), X.shape[1]))
    np.add.at(scores, np.unique(ids, return_inverse=True)[1], X * u[:, None])
    V = bread @ (scores.T @ scores) @ bread
    k = 2 + len(events)
    return b[k:], V[k:, k:]


@pytest.mark.parametrize(
    "case",
    ["balanced", "anticipation", "weights", "unbalanced", "treatment"],
)
def test_matches_the_explicit_stacked_regression(case):
    d = simulate_panel(500, 9, seed=31)
    lo, hi, A, options, weight = -3, 2, 0, {}, None
    if case == "anticipation":
        # cohort 2 has no clean period before 2 - 1
        d = d.filter(pl.col("first_treat") != 2)
        A, options = 1, {"anticipation": 1}
    if case == "weights":
        d = d.with_columns((0.5 + (pl.col("id") % 5) / 4).alias("w"))
        weight, options = "w", {"weights": "w"}
    if case == "unbalanced":
        d = d.filter(~((pl.col("id") % 7 == 0) & (pl.col("time").is_in([3, 6]))))
        options = {"incomplete_units": "keep"}
    expected_b, expected_V = regression(stack(d, lo, hi, A, weight), lo, hi, -A - 1)
    if case == "treatment":
        d = d.with_columns(
            ((pl.col("first_treat") > 0) & (pl.col("time") >= pl.col("first_treat")))
            .cast(pl.Int8)
            .alias("treated")
        )
        fit = did_stacked(
            "y ~ 1",
            d.drop("first_treat"),
            panel=("id", "time"),
            treatment="treated",
            event_window=(lo, hi),
            n_threads=2,
        )
    else:
        fit = did_stacked("y ~ 1", d, event_window=(lo, hi), **options, **KW)
    assert isinstance(fit, StackedResult)
    assert fit.coefnames == [f"event::{e}" for e in range(lo, hi + 1) if e != -A - 1]
    assert_allclose(fit.beta, expected_b, atol=1e-10, rtol=0)
    assert_allclose(fit.vcov, expected_V, atol=1e-10, rtol=0)


def test_results_and_options():
    d = simulate_panel(500, 9, seed=31)
    fit = did_stacked("y ~ 1", d, event_window=(-3, 2), **KW)
    table = fit.sub_experiments()
    assert_allclose(table["treated_share"].sum(), 1)
    included = table["cohort"].to_list()
    assert all(a - 3 >= 1 and a + 2 <= 9 for a in included)
    assert {c["cohort"] for c in fit.diagnostics["excluded_cohorts"]}.isdisjoint(included)
    post = fit.overall()
    assert post.coefnames == ["ATT"]
    assert_allclose(post.beta[0], fit.beta[-3:].mean())
    assert fit.select([0, 1]).coefnames == ["event::0", "event::1"]
    test = fit.pretrend_test()
    assert test["df1"] == 2
    assert_allclose(fit.with_vcov("CRV1").vcov, fit.vcov * 500 / 499)
    with pytest.raises(ValueError, match="reference period"):
        did_stacked("y ~ 1", d, event_window=(0, 2), **KW)
    with pytest.raises(ValueError, match="no sub-experiment"):
        did_stacked("y ~ 1", d, event_window=(-8, 8), **KW)
    with pytest.raises(ValueError, match="no covariates"):
        did_stacked("y ~ x", d, event_window=(-3, 2), **KW)
    p = did_stacked("y ~ 1", d, event_window=(-3, 2), profile=True, **KW)
    assert p.model == "diff-stream did_stacked"


def test_biennial_labels():
    d = simulate_panel(400, 8, seed=3)
    spaced = d.with_columns(
        (1999 + 2 * pl.col("time")).alias("time"),
        pl.when(pl.col("first_treat") == 0)
        .then(0)
        .otherwise(1999 + 2 * pl.col("first_treat"))
        .alias("first_treat"),
    )
    a = did_stacked("y ~ 1", d, event_window=(-2, 2), **KW)
    b = did_stacked("y ~ 1", spaced, event_window=(-2, 2), **KW)
    assert_allclose(a.beta, b.beta, atol=1e-12)
    assert set(b.sub_experiments()["cohort"].to_list()) <= set(range(2001, 2016, 2))


@pytest.mark.parametrize("case", ["balanced", "weighted", "unbalanced", "anticipation"])
def test_matches_fixest_on_the_stack(case):
    import json
    from pathlib import Path

    from benchmarks._files import digest, read_csv

    ref = Path(__file__).parent / "data/stacked_reference"
    for name, h in json.loads((ref / "PROVENANCE.json").read_text())["files"].items():
        assert digest(Path(__file__).parents[1] / name) == h, name
    d = read_csv(Path(__file__).parent / "data/stacked_panel.csv")
    windows = {
        "balanced": (-3, 2),
        "weighted": (-3, 2),
        "unbalanced": (-2, 3),
        "anticipation": (-4, 2),
    }
    options = {}
    if case == "weighted":
        options = {"weights": "w"}
    if case == "unbalanced":
        d = d.filter(~((pl.col("id") % 7 == 0) & pl.col("time").is_in([3, 6])))
        options = {"incomplete_units": "keep"}
    if case == "anticipation":
        d = d.filter(pl.col("first_treat") != 2)
        options = {"anticipation": 1}
    fit = did_stacked("y ~ 1", d, event_window=windows[case], **options, **KW)
    expected = read_csv(ref / f"{case}.csv")
    V = read_csv(ref / f"{case}_vcov.csv", has_header=False).to_numpy()
    ix = [expected["term"].to_list().index(n) for n in fit.coefnames]
    assert_allclose(fit.beta, expected["estimate"].to_numpy()[ix], atol=1e-10, rtol=0)
    assert_allclose(fit.vcov, V[np.ix_(ix, ix)], atol=1e-10, rtol=0)
