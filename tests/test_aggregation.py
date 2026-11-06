"""External aggregation references, composition arithmetic and support rules."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from benchmarks._files import read_csv
from diff_stream import diff_stream
from diff_stream.simulate import simulate_panel

ROOT = Path(__file__).parent / "data"
KW = dict(panel=("id", "time"), cohort="first_treat", threads=2)


@pytest.fixture(scope="module", params=["balanced", "unbalanced"])
def reference_sample(request):
    name = "reference_panel.csv" if request.param == "balanced" else "unbalanced_panel.csv"
    return request.param, read_csv(ROOT / name)


@pytest.mark.parametrize("control", ["nevertreated", "notyettreated"])
@pytest.mark.parametrize("base", ["varying", "universal"])
@pytest.mark.parametrize(
    "case",
    [
        "cohort",
        "calendar",
        "dynamic_balance0",
        "dynamic_balance1",
        "dynamic_balance2",
        "cohort_horizons",
    ],
)
def test_r_aggte(reference_sample, control, base, case):
    sample, data = reference_sample
    r = diff_stream(
        "y ~ 1" if sample == "balanced" else "y ~ x + z",
        data,
        control_group=control,
        base_period=base,
        panel_mode="balanced" if sample == "balanced" else "unbalanced",
        **KW,
    )
    options = {"balance_e": int(case[-1])} if "balance" in case else {}
    if case == "cohort_horizons":
        options["horizons"] = [0, 1]
    kind = case.split("_")[0]
    a = r.aggregate(kind, **options)
    ref = read_csv(
        ROOT / "aggregation_reference" / f"{sample}_{control}_{base}_{case}.csv", null_values="NA"
    )
    levels = [int(n.split("::")[-1]) for n in a.coefnames]
    ix = [ref["level"].to_list().index(n) for n in levels]
    assert_allclose(a.beta, ref["att"].to_numpy()[ix], atol=2e-8)
    assert_allclose(a.se, ref["se"].to_numpy()[ix], atol=2e-8)
    V = read_csv(
        ROOT / "aggregation_reference" / f"{sample}_{control}_{base}_{case}_vcov.csv"
    ).to_numpy()
    assert_allclose(a.vcov, V[np.ix_(ix, ix)], atol=2e-8)
    c = a.contributions()
    for j, name in enumerate(a.coefnames):
        rows = c.filter(pl.col("aggregate") == name)
        assert_allclose(rows["weight"].sum(), 1)
        assert_allclose(rows["weighted_estimate"].sum(), a.beta[j])
    assert a.contributions().equals(r.contributions(kind, **options))
    assert a.composition().equals(r.composition(kind, **options))
    if kind == "cohort":
        assert_allclose(a.beta, r.aggregate("group", **options).beta)


def dense_bjs(data, cells, A, average, leaveout):
    """Independent full-unit-dummy calculation of published smart scores."""
    ids = data["id"].to_numpy()
    t, g, y = (data[name].to_numpy() for name in ["time", "first_treat", "y"])
    units = np.unique(ids)
    X = np.column_stack([ids == u for u in units] + [t == v for v in np.unique(t)[1:]]).astype(
        float
    )
    treated = (g > 0) & (t >= g)
    pinv = np.linalg.pinv(X[~treated])
    residual = y - X @ (pinv @ y[~treated])
    w = (
        np.column_stack(
            [
                ((g == cg) & (t == ct)).astype(float) / np.sum((g == cg) & (t == ct))
                for cg, ct in cells
            ]
        )
        @ A.T
    )
    beta = w.T @ residual
    W = w.copy()
    W[~treated] = (-(X[treated].T @ w[treated]).T @ pinv).T
    label = [
        {
            "cohort_time": (int(cg), int(ct)),
            "event_time": int(ct - cg),
            "cohort": int(cg),
            "all": 0,
        }[average]
        for cg, ct in zip(g, t, strict=True)
    ]
    scores = np.zeros((len(units), len(A)))
    for j in range(len(A)):
        eps = residual.copy()
        for group in set(label[k] for k in np.flatnonzero(treated)):
            rows = np.array([k for k in np.flatnonzero(treated) if label[k] == group])
            cluster_weights = np.array([w[rows[ids[rows] == u], j].sum() for u in units])
            denom = cluster_weights @ cluster_weights
            numerator = sum(
                cluster_weights[k] * (w[rows[ids[rows] == u], j] @ residual[rows[ids[rows] == u]])
                for k, u in enumerate(units)
            )
            center = numerator / denom if denom else 0
            eps[rows] -= center
            if leaveout and denom:
                lev = np.array(
                    [cluster_weights[np.searchsorted(units, ids[k])] ** 2 / denom for k in rows]
                )
                eps[rows] /= 1 - lev
        scores[:, j] = [W[ids == u, j] @ eps[ids == u] for u in units]
    return beta, scores.T @ scores


@pytest.mark.parametrize("unbalanced", [False, True])
@pytest.mark.parametrize("average", ["cohort_time", "event_time", "cohort", "all"])
@pytest.mark.parametrize("leaveout", [False, True])
def test_new_bjs_contrasts(data, unbalanced, average, leaveout):
    p = data.filter(~((pl.col("id") % 3 == 0) & (pl.col("time") == 3))) if unbalanced else data
    r = diff_stream(
        "y ~ 1",
        p,
        estimator="bjs",
        panel_mode="unbalanced",
        avgeffectsby=average,
        leaveout=leaveout,
        **KW,
    )
    try:
        for kind in ["cohort", "calendar", "dynamic"]:
            options = {"balance_window": (0, 1)} if kind == "dynamic" else {}
            a = r.aggregate(kind, **options)
            keys = (
                [0, 1]
                if kind == "dynamic"
                else sorted({g if kind == "cohort" else t for g, t in r.cells})
            )
            A = np.zeros((len(keys), len(r.cells)))
            for j, key in enumerate(keys):
                ix = [
                    k
                    for k, (g, t) in enumerate(r.cells)
                    if (
                        (g <= int(p["time"].max()) - 1 and t - g == key)
                        if kind == "dynamic"
                        else (g if kind == "cohort" else t) == key
                    )
                ]
                counts = np.array(
                    [
                        p.filter(
                            (pl.col("first_treat") == r.cells[k][0])
                            & (pl.col("time") == r.cells[k][1])
                        ).height
                        for k in ix
                    ]
                )
                A[j, ix] = counts / counts.sum()
            beta, V = dense_bjs(p, r.cells, A, average, leaveout)
            assert_allclose(a.beta, beta, atol=1e-10)
            assert_allclose(a.vcov, V, atol=1e-10)
    finally:
        r.cleanup()


@pytest.mark.parametrize("estimator", ["cs", "sunab", "bjs"])
def test_counts_composition_and_balancing(estimator):
    data = simulate_panel(160, 7, seed=81)
    data = data.filter(~((pl.col("id") % 3 == 0) & (pl.col("time") == 4)))
    r = diff_stream("y ~ 1", data, estimator=estimator, panel_mode="unbalanced", **KW)
    try:
        original_warnings = list(r.diagnostics["warnings"])
        a = r.aggregate(balance_window=(0, 2))
        assert a.coefnames == ["event::0", "event::1", "event::2"]
        c, s = a.contributions(), a.composition()
        assert s["cohorts"].to_list()[0] == s["cohorts"].to_list()[1] == s["cohorts"].to_list()[2]
        assert not s["cohorts_changed"].any()
        assert c["cohort"].max() <= 5
        for row in c.iter_rows(named=True):
            cohort = data.filter(pl.col("first_treat") == row["cohort"])
            observed = cohort.filter(pl.col("time") == row["time"]).height
            population = cohort["id"].n_unique()
            assert row["observed_units"] == observed
            assert row["cohort_units"] == population
            assert row["weight_units"] == (population if estimator == "cs" else observed)
        if estimator == "cs":
            assert not s["cohort_weights_changed"].any()
            assert (c["observed_units"] < c["cohort_units"]).any()
        assert r.diagnostics["warnings"] == original_warnings
        b = r.with_vcov("CRV1").aggregate("cohort")
        assert_allclose(b.vcov, r.aggregate("cohort").vcov * r.n_units / (r.n_units - 1))
        assert b.contributions().equals(r.aggregate("cohort").contributions())
        r.cleanup()
        assert a.contributions().equals(c)
        assert r.composition(balance_window=(0, 2)).equals(s)
    finally:
        r.cleanup()


@pytest.mark.parametrize(
    "options",
    [
        dict(kind="bad"),
        dict(balance_e=True),
        dict(balance_e=-1),
        dict(balance_e=1.2),
        dict(balance_e=1, balance_window=(0, 1)),
        dict(kind="cohort", balance_e=1),
        dict(balance_window=(2, 0)),
        dict(balance_window=[0]),
        dict(balance_window="ab"),
        dict(horizons=[False]),
        dict(horizons=[0.5]),
        dict(horizons=2),
        dict(balance_window=(0, 1000000000)),
        dict(balance_e=1000000000),
        dict(horizons=[]),
    ],
)
def test_invalid_aggregation_options(data, options):
    r = diff_stream("y ~ 1", data, **KW)
    with pytest.raises(ValueError):
        r.aggregate(**options)


def test_support_holes_and_baseline_normalization():
    data = simulate_panel(150, 7, seed=49)
    # Remove cohort 3's entire exposure-1 cell, leaving exposures 0 and 2.
    data = data.filter(~((pl.col("first_treat") == 3) & (pl.col("time") == 4)))
    r = diff_stream("y ~ 1", data, estimator="sunab", panel_mode="unbalanced", **KW)
    a = r.aggregate(balance_window=(-1, 2))
    assert a.coefnames == ["event::0", "event::1", "event::2"]
    assert 3 not in a.contributions()["cohort"]
    assert any(
        x["cohort"] == 3 and x["missing_horizons"] == [1] for x in a.diagnostics["excluded_cohorts"]
    )
    # Balancing precedes display horizon selection, even when the hole is hidden.
    b = r.aggregate(balance_window=(0, 2), horizons=[0, 2])
    assert 3 not in b.contributions()["cohort"]
    assert not b.composition()["cohorts_changed"].any()


def test_group_time_counts_and_missing_metadata(data):
    r = diff_stream("y ~ 1", data, **KW)
    table = r.group_time()
    assert table["weight_units"].equals(table["n_units"])
    assert table["observed_units"].equals(table["cohort_units"])
    old = replace(r, observed_counts=None)
    assert old.contributions()["observed_units"].null_count() == len(old.contributions())
    a = r.aggregate("calendar")
    with pytest.raises(ValueError, match="without aggregation arguments"):
        a.contributions("cohort")
    with pytest.raises(ValueError, match="original"):
        a.aggregate()


@pytest.mark.parametrize("unbalanced", [False, True])
def test_sa_aggregates_dense_regression(data, unbalanced):
    from oracles import regression

    p = data.filter(~((pl.col("id") % 3 == 0) & (pl.col("time") == 3))) if unbalanced else data
    r = diff_stream("y ~ 1", p, estimator="sunab", panel_mode="unbalanced", **KW)
    beta, V = regression(p, "sunab", r.cells)
    for kind in ["cohort", "calendar"]:
        keys = sorted({g if kind == "cohort" else t for g, t in r.cells if t >= g})
        A = np.zeros((len(keys), len(r.cells)))
        for j, key in enumerate(keys):
            selected = np.array(
                [t >= g and (g if kind == "cohort" else t) == key for g, t in r.cells]
            )
            weights = (
                np.array(
                    [
                        p.filter((pl.col("first_treat") == g) & (pl.col("time") == t)).height
                        for g, t in r.cells
                    ]
                )
                * selected
            )
            A[j] = weights / weights.sum()
        a = r.aggregate(kind)
        assert_allclose(a.beta, A @ beta, atol=1e-10)
        assert_allclose(a.vcov, A @ V @ A.T, atol=1e-10)


@pytest.mark.parametrize("base,cohorts", [("varying", [4, 5]), ("universal", [3, 4, 5])])
def test_prepost_windows_respect_normalization(base, cohorts):
    data = simulate_panel(200, 7, seed=83)
    r = diff_stream("y ~ 1", data, base_period=base, **KW)
    a = r.aggregate(balance_window=(-2, 2))
    assert all(c == cohorts for c in a.composition()["cohorts"].to_list())
    assert (-1 in a.contributions()["event_time"]) == (base == "varying")
    assert not a.diagnostics["cohort_composition_changes"]


def test_balance_e_guarantees_post_membership_only(data):
    r = diff_stream("y ~ 1", data, **KW)
    a = r.aggregate(balance_e=1)
    rows = a.composition().filter(pl.col("aggregate").is_in(["event::0", "event::1"]))
    assert rows["cohorts"].to_list()[0] == rows["cohorts"].to_list()[1]
    assert a.diagnostics["cohort_composition_changes"]
    b = r.aggregate(balance_e=1, horizons=[0, 1])
    assert not b.diagnostics["cohort_composition_changes"]
    assert not b.diagnostics["cohort_weights_change"]


def test_calendar_horizons_select_cells(data):
    r = diff_stream("y ~ 1", data, **KW)
    a = r.aggregate("calendar", horizons=[0])
    ix = [j for j, (g, t) in enumerate(r.cells) if g == t]
    assert a.coefnames == [f"calendar::{r.cells[j][0]}" for j in ix]
    assert_allclose(a.beta, r.beta[ix], atol=1e-12)
    assert_allclose(a.vcov, r.vcov[np.ix_(ix, ix)], atol=1e-12)
    assert a.contributions()["weight"].to_list() == [1.0] * len(ix)


def test_cohort_without_any_post_cells_is_reported(data):
    p = data.filter(~((pl.col("first_treat") == 5) & (pl.col("time") == 5)))
    r = diff_stream("y ~ 1", p, estimator="bjs", panel_mode="unbalanced", **KW)
    try:
        a = r.aggregate("cohort")
        assert 5 not in a.contributions()["cohort"]
        assert any(x["cohort"] == 5 for x in a.diagnostics["excluded_cohorts"])
        b = r.aggregate(balance_e=0)
        assert any(
            x["cohort"] == 5 and x["missing_horizons"] == [0]
            for x in b.diagnostics["excluded_cohorts"]
        )
        with pytest.raises(ValueError, match="post-treatment"):
            r.aggregate(balance_window=(-1, 1))
    finally:
        r.cleanup()
