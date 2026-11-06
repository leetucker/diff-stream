"""Time-varying sampling weights and covariate/crossed-FE regression checks."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from benchmarks._files import read_csv
from diff_stream import diff_stream

ROOT = Path(__file__).parent / "data"
REF = ROOT / "richer_reference"
KW = dict(panel=("id", "time"), cohort="first_treat", threads=2)
SPECS = ["plain_varying", "numeric_varying", "extra_fe_varying", "extra_fe_unit", "extra_fe_equal"]


def configuration(spec):
    fml = "y ~ 1" if spec == "plain_varying" else "y ~ x + I(x**2)"
    if spec.startswith("extra_fe"):
        fml += " | id + time + region^time"
    weight = None if spec == "extra_fe_equal" else "unit_w" if spec == "extra_fe_unit" else "w"
    return fml, weight


@pytest.fixture(scope="module", params=["balanced", "unbalanced"])
def sample(request):
    name = "richer_panel.csv" if request.param == "balanced" else "richer_unbalanced_panel.csv"
    return request.param, read_csv(ROOT / name)


def check_reference(result, prefix, atol=1e-9):
    ref = read_csv(REF / (prefix + ".csv"))
    keys = ref["term"].to_list()
    ix = [keys.index(c) for c in result.coefnames]
    assert_allclose(result.beta, ref["estimate"].to_numpy()[ix], atol=atol, rtol=0)
    V = read_csv(REF / (prefix + "_vcov.csv")).to_numpy()
    assert_allclose(result.vcov, V[np.ix_(ix, ix)], atol=atol, rtol=0)


@pytest.mark.parametrize("A", [0, 1])
@pytest.mark.parametrize("design", ["constant", "numeric"])
@pytest.mark.parametrize("method", ["reg", "ipw", "dr"])
@pytest.mark.parametrize("control", ["nevertreated", "notyettreated"])
@pytest.mark.parametrize("base", ["varying", "universal"])
def test_actual_did_time_varying_weights(sample, A, design, method, control, base):
    label, d = sample
    r = diff_stream(
        "y ~ 1" if design == "constant" else "y ~ x+z",
        d,
        weights="w",
        allow_time_varying_weights=True,
        panel_mode=label,
        anticipation=A,
        est_method=method,
        control_group=control,
        base_period=base,
        ess_warn_threshold=None,
        **KW,
    )
    prefix = f"cs_{label}_{A}_{design}_{control}_{base}_{method}"
    check_reference(r, prefix, atol=3e-8)
    for kind in ["dynamic", "simple", "cohort", "calendar"]:
        if base == "varying" and kind != "dynamic":
            continue
        suffix = "group" if kind == "cohort" else kind
        check_reference(r.aggregate(kind), prefix + "_" + suffix, atol=3e-8)
    assert not r.diagnostics["weights_time_invariant"]
    assert r.diagnostics["unit_weight_anchor"].startswith(
        "first" if label == "balanced" else "mean"
    )


@pytest.mark.parametrize("A", [0, 1])
@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("estimator", ["sunab", "bjs"])
def test_r_joint_regressions_and_clean_placebos(sample, A, spec, estimator, tmp_path):
    label, d = sample
    fml, weight = configuration(spec)
    r = diff_stream(
        fml,
        d,
        estimator=estimator,
        weights=weight,
        allow_time_varying_weights=True,
        panel_mode=label,
        anticipation=A,
        pretrends=2 if estimator == "bjs" else 0,
        workdir=tmp_path,
        batch_units=37,
        **KW,
    )
    try:
        prefix = f"{estimator}_{label}_{A}_{spec}"
        if estimator == "sunab":
            check_reference(r, prefix)
        else:
            for kind in ["cells", "dynamic", "simple"]:
                check_reference(
                    r if kind == "cells" else r.aggregate(kind),
                    prefix + "_cohort_time_plugin_" + kind,
                )
            check_reference(r.placebos(), f"placebo_{label}_{A}_{spec}")
    finally:
        r.cleanup()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("A", [0, 1])
@pytest.mark.parametrize("average", ["cohort_time", "event_time", "cohort", "all"])
@pytest.mark.parametrize("leaveout", [False, True])
def test_r_dense_richer_bjs_smart_scores(sample, A, average, leaveout, tmp_path):
    label, d = sample
    fml, weight = configuration("extra_fe_varying")
    r = diff_stream(
        fml,
        d,
        estimator="bjs",
        weights=weight,
        allow_time_varying_weights=True,
        anticipation=A,
        panel_mode=label,
        avgeffectsby=average,
        leaveout=leaveout,
        workdir=tmp_path,
        **KW,
    )
    try:
        for kind in ["cells", "dynamic", "simple"]:
            a = r if kind == "cells" else r.aggregate(kind)
            prefix = f"bjs_{label}_{A}_extra_fe_varying_{average}_{'leaveout' if leaveout else 'plugin'}_{kind}"
            check_reference(a, prefix)
    finally:
        r.cleanup()
    assert not list(tmp_path.iterdir())


def dense_oracle(d, cells, estimator, *, anticipation=0, contrasts=None, leaveout=False):
    """Explicit original-unit dummies, no production elimination/design helpers."""
    u, t, g, y, w = (d[c].to_numpy() for c in ["id", "time", "first_treat", "y", "w"])
    region, x = d["region"].to_numpy(), d["x"].to_numpy()
    X = np.column_stack(
        [u == i for i in np.unique(u)]
        + [t == z for z in np.unique(t)[1:]]
        + [x, x * x]
        + [(region == rg) & (t == tt) for rg in np.unique(region)[1:] for tt in np.unique(t)[1:]]
    ).astype(float)
    N, M = len(np.unique(u)), len(cells)
    obs = (g > 0) & (t >= g - anticipation)
    if estimator == "sunab":
        X = np.column_stack([X] + [(g == cg) & (t == ct) for cg, ct in cells])
        H = np.linalg.solve(X.T @ (w[:, None] * X), np.eye(X.shape[1])[:, -M:])
        beta = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * y))
        residual = y - X @ beta
        rows = w[:, None] * residual[:, None] * (X @ H)
        scores = np.zeros((N, M))
        np.add.at(scores, u, rows)
        return beta[-M:], scores.T @ scores
    Q = X[~obs].T @ (w[~obs, None] * X[~obs])
    beta = np.linalg.solve(Q, X[~obs].T @ (w[~obs] * y[~obs]))
    residual = y - X @ beta
    W = np.column_stack([(g == cg) & (t == ct) for cg, ct in cells]) * w[:, None]
    W /= W.sum(axis=0)
    A = np.eye(M) if contrasts is None else np.atleast_2d(contrasts)
    W = W @ A.T
    rows = np.zeros_like(W)
    rows[~obs] = (
        -w[~obs, None] * residual[~obs, None] * (X[~obs] @ np.linalg.solve(Q, X[obs].T @ W[obs]))
    )
    # All-treated smart grouping: independently sum weights/effects by unit.
    for j in range(W.shape[1]):
        total, effect = np.zeros(N), np.zeros(N)
        np.add.at(total, u[obs], W[obs, j])
        np.add.at(effect, u[obs], W[obs, j] * residual[obs])
        denom = total @ total
        average = total @ effect / denom
        eps = residual[obs] - average
        if leaveout:
            eps /= 1 - total[u[obs]] ** 2 / denom
        rows[obs, j] = W[obs, j] * eps
    scores = np.zeros((N, W.shape[1]))
    np.add.at(scores, u, rows)
    return W.T @ residual, scores.T @ scores


@pytest.mark.parametrize("estimator", ["sunab", "bjs"])
def test_independent_python_full_dummy_oracle(sample, estimator):
    label, d = sample
    r = diff_stream(
        configuration("extra_fe_varying")[0],
        d,
        estimator=estimator,
        weights="w",
        allow_time_varying_weights=True,
        panel_mode=label,
        avgeffectsby="all" if estimator == "bjs" else "cohort_time",
        leaveout=estimator == "bjs",
        **KW,
    )
    try:
        if estimator == "sunab":
            beta, V = dense_oracle(d, r.cells, estimator)
            assert_allclose(r.beta, beta, atol=1e-10, rtol=0)
            assert_allclose(r.vcov, V, atol=1e-10, rtol=0)
        else:
            a = r.aggregate("simple")
            A = r.counts / r.counts.sum()
            beta, V = dense_oracle(d, r.cells, estimator, contrasts=A, leaveout=True)
            assert_allclose(a.beta, beta, atol=1e-10, rtol=0)
            assert_allclose(a.vcov, V, atol=1e-10, rtol=0)
    finally:
        r.cleanup()


@pytest.mark.parametrize("estimator", ["cs", "sunab", "bjs"])
def test_scale_layout_complete_and_crv_invariance(sample, estimator, tmp_path):
    label, d = sample
    formula = "y ~ x+z" if estimator == "cs" else configuration("extra_fe_varying")[0]
    options = dict(
        estimator=estimator,
        weights="w",
        allow_time_varying_weights=True,
        panel_mode=label,
        ess_warn_threshold=None,
        pretrends=2 if estimator == "bjs" else 0,
    )
    a = diff_stream(formula, d, **options, **KW)
    try:
        for scale in [1e-150, 1e150]:
            source = d.with_columns(
                (pl.col("w") * scale).alias("w"),
                ("person_" + pl.col("id").cast(pl.String)).alias("id"),
            ).sample(fraction=1, shuffle=True, seed=23)
            path = tmp_path / "source.parquet"
            source.write_parquet(path, row_group_size=13)
            b = diff_stream(formula, str(path), preprocess_mb=1, batch_units=17, **options, **KW)
            try:
                assert_allclose(a.beta, b.beta, atol=1e-9, rtol=0)
                assert_allclose(a.vcov, b.vcov, atol=1e-9, rtol=0)
                assert_allclose(
                    a.aggregate("simple").vcov, b.aggregate("simple").vcov, atol=1e-9, rtol=0
                )
                assert_allclose(
                    b.with_vcov("CRV1").aggregate("simple").vcov,
                    b.aggregate("simple").vcov * b.n_units / (b.n_units - 1),
                    atol=1e-11,
                )
                if estimator == "bjs":
                    assert_allclose(a.placebos().vcov, b.placebos().vcov, atol=1e-9, rtol=0)
            finally:
                b.cleanup()
    finally:
        a.cleanup()
    complete = d.filter(pl.len().over("id") == d["time"].n_unique())
    a = diff_stream(formula, d, **dict(options, panel_mode="complete"), **KW)
    b = diff_stream(formula, complete, **dict(options, panel_mode="balanced"), **KW)
    try:
        assert_allclose(a.beta, b.beta, atol=1e-10, rtol=0)
        assert_allclose(a.vcov, b.vcov, atol=1e-10, rtol=0)
    finally:
        a.cleanup()
        b.cleanup()


@pytest.mark.parametrize("estimator", ["sunab", "bjs"])
def test_absorbed_covariates_scaling_and_identification(estimator, tmp_path):
    d = read_csv(ROOT / "richer_panel.csv")
    f = configuration("extra_fe_varying")[0]
    kw = dict(estimator=estimator, weights="w", allow_time_varying_weights=True, **KW)
    a = diff_stream(f, d, **kw)
    b = diff_stream(
        f.replace("I(x**2)", "I(x**2) + z"), d.with_columns((pl.col("x") * 1e12).alias("x")), **kw
    )
    try:
        assert "z" in b.diagnostics["absorbed_or_redundant_columns"]
        assert_allclose(a.beta, b.beta, atol=1e-9, rtol=0)
        assert_allclose(a.vcov, b.vcov, atol=1e-9, rtol=0)
    finally:
        a.cleanup()
        b.cleanup()
    with pytest.raises(ValueError, match="unidentified|singular"):
        diff_stream("y ~ x | first_treat^time", d, workdir=tmp_path, **kw)
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError, match="unit-interacted"):
        diff_stream("y ~ x | id^region", d, **kw)
    with pytest.raises(ValueError, match="column names"):
        diff_stream("y ~ x | region[x]", d, **kw)


@pytest.mark.parametrize("estimator", ["cs", "sunab", "bjs"])
def test_time_varying_weights_require_explicit_opt_in(estimator):
    d = read_csv(ROOT / "richer_panel.csv")
    with pytest.raises(ValueError, match="constant within unit"):
        diff_stream("y ~ 1", d, estimator=estimator, weights="w", **KW)
    with pytest.raises(ValueError, match="must be boolean"):
        diff_stream("y ~ 1", d, estimator=estimator, allow_time_varying_weights="yes", **KW)


def test_varying_cs_multiplier_and_ess(sample):
    label, d = sample
    r = diff_stream(
        "y ~ x+z",
        d,
        panel_mode=label,
        weights="w",
        allow_time_varying_weights=True,
        bootstrap_reps=19,
        bootstrap_seed=8,
        ess_warn_threshold=None,
        **KW,
    )
    assert r.effective_sample_size()["effective_n"].min() > 0
    assert np.isfinite(r._multiplier_draws).all()
    assert np.isfinite(r.aggregate("simple")._multiplier_draws).all()
    assert np.isfinite(r.multiplier_confint().drop("Coefficient").to_numpy()).all()


def test_reference_provenance():
    from benchmarks.richer_validation import provenance

    provenance()


@pytest.mark.parametrize("estimator", ["sunab", "bjs"])
def test_redundant_extra_fe_preserves_specialized_fit(estimator):
    d = read_csv(ROOT / "richer_panel.csv")
    kw = dict(estimator=estimator, weights="unit_w", **KW)
    a = diff_stream("y ~ 1", d, **kw)
    b = diff_stream("y ~ 1 | id + time + region", d, **kw)
    try:
        assert_allclose(a.beta, b.beta, atol=1e-10, rtol=0)
        assert_allclose(a.vcov, b.vcov, atol=1e-10, rtol=0)
        assert any("region" in s for s in b.diagnostics["absorbed_or_redundant_columns"])
    finally:
        a.cleanup()
        b.cleanup()


def test_future_cohort_timing_and_richer_placebo_support(sample):
    label, d = sample
    d = d.with_columns(
        pl.when(pl.col("id") % 7 == 0).then(9).otherwise(pl.col("first_treat")).alias("first_treat")
    )
    r = diff_stream(
        configuration("extra_fe_varying")[0],
        d,
        estimator="bjs",
        weights="w",
        allow_time_varying_weights=True,
        panel_mode=label,
        anticipation=1,
        pretrends=2,
        avgeffectsby="all",
        **KW,
    )
    try:
        assert all(g != 9 for g, t in r.cells)
        assert 9 in r.placebos().support()["cohort"]
        a = r.aggregate("simple")
        A = r.counts * np.array([t >= g for g, t in r.cells])
        A /= A.sum()
        beta, V = dense_oracle(d, r.cells, "bjs", anticipation=1, contrasts=A)
        assert_allclose(a.beta, beta, atol=1e-10, rtol=0)
        assert_allclose(a.vcov, V, atol=1e-10, rtol=0)
    finally:
        r.cleanup()
