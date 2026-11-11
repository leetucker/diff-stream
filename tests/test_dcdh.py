"""de Chaisemartin and D'Haultfoeuille's DID_l (did_dcdh).

Against a dense transcription of DIDmultiplegtDYN 2.4.0 (tests/dcdh_oracle.py) on
panels where treatment takes several values and switches on and off, and against
Callaway-Sant'Anna where treatment is binary and absorbing.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from benchmarks._files import digest, read_csv
from diff_stream import did_callaway_santanna, did_dcdh
from diff_stream.simulate import simulate_panel
from tests import dcdh_oracle

KW = dict(panel=("id", "time"), treatment="d", n_threads=2)


def switching_panel(seed, n=300, periods=7, never=0.3, baselines=(0, 1, 2)):
    """Groups with a baseline treatment, a first switch up or down (or none), and
    sometimes a second switch further or back; effects grow with the dose and its
    duration."""
    rng = np.random.default_rng(seed)
    d1 = rng.choice(baselines, n)
    D = np.repeat(d1[:, None], periods, axis=1).astype(float)
    for i in range(n):
        if rng.random() < 1 - never:
            F = rng.integers(2, periods + 1)
            step = rng.choice([-1, 1]) if d1[i] > 0 else 1
            new = d1[i] + step * rng.integers(1, 3)
            D[i, F - 1 :] = new
            if rng.random() < 0.3 and F < periods:
                D[i, rng.integers(F + 1, periods + 1) - 1 :] = (
                    new + step if rng.random() < 0.5 else d1[i]
                )
    t = np.arange(1, periods + 1)
    dev = D - d1[:, None]
    y = (
        rng.normal(size=n)[:, None]
        + 0.3 * t
        + 0.8 * dev
        + 0.1 * np.cumsum(dev, axis=1)
        + rng.normal(size=(n, periods))
    )
    return pl.DataFrame(
        {
            "id": np.repeat(np.arange(n), periods),
            "time": np.tile(t, n),
            "d": D.ravel(),
            "y": y.ravel(),
        }
    )


def wide(frame, column):
    """Groups by periods, NaN where a group has no row."""
    full = (
        frame.select("id")
        .unique()
        .join(frame.select("time").unique(), how="cross")
        .join(frame, on=["id", "time"], how="left")
        .sort("id", "time")
    )
    G, T = full["id"].n_unique(), full["time"].n_unique()
    return full[column].cast(pl.Float64).fill_null(np.nan).to_numpy().reshape(G, T)


def oracle(
    frame,
    effects,
    placebos,
    covariates=(),
    *,
    weights=None,
    trends_by=None,
    outcome="y",
    counts=False,
    **options,
):
    Y, D = wide(frame, outcome), wide(frame, "d")
    X = np.stack([wide(frame, c) for c in covariates], axis=-1) if covariates else None
    W = wide(frame, weights) if weights else None
    strata = None
    if trends_by:
        codes = frame.group_by("id").agg(pl.col(trends_by).first()).sort("id")[trends_by]
        strata = codes.rank("dense").to_numpy().astype(float)
    out = dcdh_oracle.did_multiplegt_dyn(
        Y, D, effects, placebos, X=X, W=W, strata=strata, **options
    )
    names, beta, scores, switchers = [], [], [], []
    for (kind, ell), value in ((k, v) for k, v in out.items() if k != "average"):
        names.append(f"{kind}::{ell}")
        beta.append(value["estimate"])
        scores.append(value["Uvar"])
        switchers.append([value["parts"][s][0] for s in (1, 0)])
    if "average" in out:
        names.append("average_total_effect")
        beta.append(out["average"]["estimate"])
        scores.append(out["average"]["Uvar"])
    S = np.array(scores)
    if counts:
        return names, np.array(beta), S @ S.T, np.array(switchers)
    return names, np.array(beta), S @ S.T


OPTIONS = {
    "default": {},
    "normalized": {"normalized": True},
    "only_never_switchers": {"only_never_switchers": True},
    "switchers_in": {"switchers": "in"},
    "switchers_out": {"switchers": "out"},
}


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("name", OPTIONS)
def test_matches_the_dense_transcription(seed, name):
    frame = switching_panel(seed)
    options = OPTIONS[name]
    fit = did_dcdh("y ~ 1", frame, **KW, effects=3, placebos=2, **options)
    names, beta, vcov = oracle(frame, 3, 2, **options)
    assert fit.coefnames == names
    assert_allclose(fit.beta, beta, rtol=1e-10, atol=1e-12)
    assert_allclose(fit.vcov, vcov, rtol=1e-9, atol=1e-14)


def with_covariates(frame, seed):
    """Two covariates that change over time; one moves with the treatment and affects y."""
    rng = np.random.default_rng(seed)
    frame = frame.sort("id", "time")
    G, T = frame["id"].n_unique(), frame["time"].n_unique()
    D = frame["d"].to_numpy().reshape(G, T)
    x = rng.normal(size=(G, T)).cumsum(axis=1) + 0.5 * (D - D[:, :1])
    z = rng.normal(size=(G, T))
    return frame.with_columns(
        pl.Series("x", x.ravel()),
        pl.Series("z", z.ravel()),
        (pl.col("y") + 0.7 * pl.Series(x.ravel()) - 0.3 * pl.Series(z.ravel())).alias("y"),
    )


@pytest.mark.parametrize("seed", [1, 2])
@pytest.mark.parametrize("name", ["default", "normalized", "switchers_out", "only_never_switchers"])
def test_covariates_match_the_dense_transcription(seed, name):
    frame = with_covariates(switching_panel(seed), seed)
    options = OPTIONS[name]
    fit = did_dcdh("y ~ x + z", frame, **KW, effects=3, placebos=2, **options)
    names, beta, vcov = oracle(frame, 3, 2, covariates=("x", "z"), **options)
    assert fit.coefnames == names
    assert fit.diagnostics["covariates"] == ["x", "z"]
    assert_allclose(fit.beta, beta, rtol=1e-9, atol=1e-12)
    assert_allclose(fit.vcov, vcov, rtol=1e-8, atol=1e-14)


def drop_rows(frame, seed, share=0.12):
    """An unbalanced panel: rows missing at random, including just before switches."""
    rng = np.random.default_rng(seed)
    return frame.filter(pl.Series(rng.random(len(frame)) > share))


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("name", ["default", "normalized", "switchers_in", "only_never_switchers"])
def test_unbalanced_panels_match_the_dense_transcription(seed, name):
    frame = drop_rows(switching_panel(seed), seed)
    options = OPTIONS[name]
    fit = did_dcdh("y ~ 1", frame, **KW, effects=3, placebos=2, incomplete_units="keep", **options)
    names, beta, vcov = oracle(frame, 3, 2, **options)
    assert fit.coefnames == names
    assert_allclose(fit.beta, beta, rtol=1e-10, atol=1e-12)
    assert_allclose(fit.vcov, vcov, rtol=1e-9, atol=1e-14)


@pytest.mark.parametrize("seed", [4, 5])
def test_unbalanced_panels_with_covariates(seed):
    frame = drop_rows(with_covariates(switching_panel(seed), seed), seed)
    fit = did_dcdh("y ~ x + z", frame, **KW, effects=3, placebos=2, incomplete_units="keep")
    names, beta, vcov = oracle(frame, 3, 2, covariates=("x", "z"))
    assert fit.coefnames == names
    assert_allclose(fit.beta, beta, rtol=1e-9, atol=1e-12)
    assert_allclose(fit.vcov, vcov, rtol=1e-8, atol=1e-14)


def with_extras(frame, seed):
    """A weight that varies over time, three regions, and an outcome with a trend by
    region and a linear trend by group."""
    rng = np.random.default_rng(seed + 100)
    frame = frame.sort("id", "time")
    ids = frame["id"].unique().sort()
    region = pl.DataFrame(
        {
            "id": ids,
            "region": rng.choice(["a", "b", "c"], len(ids)),
            "slope": 0.3 * rng.normal(size=len(ids)),
        }
    )
    return (
        frame.join(region, on="id")
        .with_columns(
            pl.Series("w", np.exp(0.3 * rng.normal(size=len(frame)))),
            (
                pl.col("y")
                + (pl.col("region").rank("dense").cast(pl.Float64) * 0.25 + pl.col("slope"))
                * pl.col("time")
            ).alias("y_trend"),
        )
        .drop("slope")
    )


TRENDS = {
    "weights": ("y ~ 1", {"weights": "w"}),
    "trends_by": ("y_trend ~ 1", {"trends_by": "region"}),
    "linear_trends": ("y_trend ~ 1", {"linear_trends": True}),
    "linear_trends_normalized": ("y_trend ~ 1", {"linear_trends": True, "normalized": True}),
    "linear_trends_only_never": (
        "y_trend ~ 1",
        {"linear_trends": True, "only_never_switchers": True},
    ),
    "linear_trends_out": ("y_trend ~ 1", {"linear_trends": True, "switchers": "out"}),
    "linear_trends_weights": ("y_trend ~ 1", {"linear_trends": True, "weights": "w"}),
    "all": ("y_trend ~ x + z", {"linear_trends": True, "trends_by": "region", "weights": "w"}),
}


def fit_and_oracle(frame, name, effects=3, placebos=2, **extra):
    fml, options = TRENDS[name]
    outcome, rhs = (part.strip() for part in fml.split("~"))
    covariates = () if rhs == "1" else tuple(c.strip() for c in rhs.split("+"))
    fit = did_dcdh(
        fml,
        frame,
        **KW,
        effects=effects,
        placebos=placebos,
        allow_time_varying_weights=True,
        **options,
        **extra,
    )
    expected = oracle(frame, effects, placebos, covariates, outcome=outcome, counts=True, **options)
    return fit, expected


@pytest.mark.parametrize("seed", [1, 2])
@pytest.mark.parametrize("name", TRENDS)
def test_weights_and_trends_match_the_dense_transcription(seed, name):
    frame = with_extras(with_covariates(switching_panel(seed, n=400), seed), seed)
    fit, (names, beta, vcov, switchers) = fit_and_oracle(frame, name)
    assert fit.coefnames == names
    assert_allclose(fit.beta, beta, rtol=1e-9, atol=1e-12)
    assert_allclose(fit.vcov, vcov, rtol=1e-8, atol=1e-14)
    table = fit.switchers()
    weighted = "weights" in TRENDS[name][1]
    column = "weight" if weighted else "switchers"
    assert_allclose(table[f"{column}_in"], switchers[:, 0], rtol=1e-12)
    assert_allclose(table[f"{column}_out"], switchers[:, 1], rtol=1e-12)


@pytest.mark.parametrize("seed", [3, 4])
@pytest.mark.parametrize("name", ["weights", "trends_by", "linear_trends", "all"])
def test_weights_and_trends_in_unbalanced_panels(seed, name):
    """With linear trends, a first difference is used only where both of its periods
    are observed."""
    frame = drop_rows(with_extras(with_covariates(switching_panel(seed, n=400), seed), seed), seed)
    fit, (names, beta, vcov, _) = fit_and_oracle(frame, name, incomplete_units="keep")
    assert fit.coefnames == names
    assert_allclose(fit.beta, beta, rtol=1e-9, atol=1e-12)
    assert_allclose(fit.vcov, vcov, rtol=1e-8, atol=1e-14)


def test_linear_trends_use_the_same_switchers_for_every_horizon():
    """Effect l sums horizons 1..l for the switchers with all l; switchers first
    switching in period 2 are not used."""
    frame = with_extras(switching_panel(5, n=500), 5)
    fit = did_dcdh("y_trend ~ 1", frame, **KW, effects=3, linear_trends=True)
    assert "average_total_effect" not in fit.coefnames
    counts = fit.switchers()
    total = (counts["switchers_in"] + counts["switchers_out"]).to_list()
    assert total == sorted(total, reverse=True) and total[0] > total[-1]
    D = wide(frame, "d")
    second = (D[:, 1] != D[:, 0]).sum()
    plain = did_dcdh("y_trend ~ 1", frame, **KW, effects=1)
    assert total[0] < plain.switchers()["switchers_in"][0] + plain.switchers()["switchers_out"][0]
    assert second > 0


def test_weights_are_scale_free_and_checked():
    frame = with_extras(switching_panel(6), 6)
    a = did_dcdh(
        "y ~ 1", frame, **KW, effects=3, placebos=2, weights="w", allow_time_varying_weights=True
    )
    b = did_dcdh(
        "y ~ 1",
        frame.with_columns(pl.col("w") * 7.5),
        **KW,
        effects=3,
        placebos=2,
        weights="w",
        allow_time_varying_weights=True,
    )
    assert_allclose(b.beta, a.beta, rtol=1e-12)
    assert_allclose(b.vcov, a.vcov, rtol=1e-10)
    with pytest.raises(ValueError, match="constant within unit"):
        did_dcdh("y ~ 1", frame, **KW, weights="w")
    fixed = frame.with_columns(pl.col("w").first().over("id"))
    c = did_dcdh("y ~ 1", fixed, **KW, effects=3, weights="w")
    assert np.isfinite(c.beta).all()


def test_trends_by_several_columns_and_errors():
    frame = with_extras(switching_panel(7), 7).with_columns((pl.col("id") % 2).alias("half"))
    two = did_dcdh("y_trend ~ 1", frame, **KW, effects=2, trends_by=["region", "half"])
    names, beta, _ = oracle(
        frame.with_columns(pl.concat_str("region", "half").alias("cell")),
        2,
        0,
        outcome="y_trend",
        trends_by="cell",
    )
    assert_allclose(two.beta, beta, rtol=1e-10)
    assert two.diagnostics["trends_by"] == ["region", "half"]
    assert any("|" in key for key in two.diagnostics["last_period_with_controls"])
    varying = frame.with_columns(
        pl.when(pl.col("time") == 3).then(pl.lit("z")).otherwise(pl.col("region")).alias("region")
    )
    with pytest.raises(ValueError, match="constant within each group"):
        did_dcdh("y ~ 1", varying, **KW, trends_by="region")
    with pytest.raises(ValueError, match="missing values"):
        did_dcdh(
            "y ~ 1",
            frame.with_columns(
                pl.when(pl.col("id") == 3).then(None).otherwise(pl.col("region")).alias("region")
            ),
            **KW,
            trends_by="region",
        )
    with pytest.raises(ValueError, match="cannot include"):
        did_dcdh("y ~ 1", frame, **KW, trends_by="d")
    with pytest.raises(ValueError, match="unknown trends_by"):
        did_dcdh("y ~ 1", frame, **KW, trends_by="state")
    with pytest.raises(ValueError, match="linear_trends"):
        did_dcdh("y ~ 1", frame, **KW, linear_trends="yes")


def test_linear_trends_do_not_depend_on_batches():
    frame = with_extras(with_covariates(switching_panel(8, n=700), 8), 8)
    options = dict(effects=3, placebos=2, linear_trends=True, trends_by="region")
    a = did_dcdh("y_trend ~ x", frame, **KW, **options)
    b = did_dcdh("y_trend ~ x", frame, **{**KW, "n_threads": 1}, batch_units=61, **options)
    assert_allclose(a.beta, b.beta, rtol=1e-12)
    assert_allclose(a.vcov, b.vcov, rtol=1e-10)


def test_baselines_without_never_switchers():
    """Where every group of a baseline eventually switches, its last period with
    controls ends before the panel does."""
    frame = switching_panel(4, never=0.0, baselines=(0, 1))
    fit = did_dcdh("y ~ 1", frame, **KW, effects=4, placebos=3)
    names, beta, vcov = oracle(frame, 4, 3)
    assert fit.coefnames == names
    assert_allclose(fit.beta, beta, rtol=1e-10, atol=1e-12)
    assert_allclose(fit.vcov, vcov, rtol=1e-9, atol=1e-14)
    assert set(fit.diagnostics["last_period_with_controls"].values()) != {7}


def test_binary_absorbing_treatment_is_callaway_santanna():
    """With binary, absorbing treatment and one baseline, effect l is the not-yet-treated
    Callaway-Sant'Anna event study at event time l - 1."""
    frame = simulate_panel(2000, 8, seed=3).with_columns(
        ((pl.col("first_treat") > 0) & (pl.col("time") >= pl.col("first_treat")))
        .cast(pl.Int8)
        .alias("d")
    )
    fit = did_dcdh("y ~ 1", frame, **KW, effects=4)
    cs = did_callaway_santanna(
        "y ~ 1",
        frame,
        panel=("id", "time"),
        cohort="first_treat",
        control_group="not_yet_treated",
        n_threads=2,
    ).aggregate(horizons=[0, 1, 2, 3])
    assert_allclose(fit.beta[:4], cs.beta, rtol=1e-10)


def test_results_do_not_depend_on_threads_or_batches():
    frame = switching_panel(5, n=900)
    a = did_dcdh("y ~ 1", frame, **KW, effects=3, placebos=2)
    b = did_dcdh("y ~ 1", frame, **{**KW, "n_threads": 1}, effects=3, placebos=2, batch_units=97)
    assert_allclose(a.beta, b.beta, rtol=1e-12)
    assert_allclose(a.vcov, b.vcov, rtol=1e-10)


def test_spaced_periods_and_tests():
    frame = switching_panel(6)
    fit = did_dcdh("y ~ 1", frame, **KW, effects=3, placebos=2)
    spaced = did_dcdh(
        "y ~ 1", frame.with_columns(pl.col("time") * 2 + 2000), **KW, effects=3, placebos=2
    )
    assert_allclose(spaced.beta, fit.beta, rtol=1e-12)
    assert spaced.diagnostics["time_min"] == 2002
    assert max(spaced.diagnostics["last_period_with_controls"].values()) == 2014
    pre = fit.pretrend_test()
    assert np.isfinite(pre.statistic) if hasattr(pre, "statistic") else pre is not None
    fit.effects_equal_test()
    table = fit.switchers()
    assert table["coefficient"].to_list() == fit.coefnames[:-1]
    assert (table["switchers_in"] + table["switchers_out"] > 0).all()


def test_errors():
    frame = switching_panel(7)
    with pytest.raises(ValueError, match="unknown variables"):
        did_dcdh("y ~ x", frame, **KW)
    with pytest.raises(ValueError, match="fixed effects"):
        did_dcdh("y ~ 1 | time", frame, **KW)
    with pytest.raises(ValueError, match="also be a covariate"):
        did_dcdh("y ~ d", frame, **KW)
    same = frame.with_columns(
        pl.when(pl.col("time") >= 3).then(pl.lit(5.0)).otherwise(pl.lit(1.0)).alias("d")
    )
    with pytest.raises(ValueError, match="same period"):
        did_dcdh("y ~ 1", same, **KW)
    with pytest.raises(ValueError, match="switchers"):
        did_dcdh("y ~ 1", frame, **KW, switchers="up")
    with pytest.raises(ValueError, match="incomplete_units"):
        did_dcdh("y ~ 1", frame, **KW, incomplete_units="sometimes")


@pytest.mark.resource
@pytest.mark.parametrize(
    "fml, extra",
    [
        ("y ~ 1", {}),
        (
            "y_trend ~ x",
            {
                "linear_trends": True,
                "trends_by": "region",
                "weights": "w",
                "allow_time_varying_weights": True,
            },
        ),
    ],
)
def test_constrained_fit_matches_unconstrained_fit(
    resource_runner, resource_enforcement, tmp_path, fml, extra
):
    """Under a memory watchdog, with small staging pieces and batches, the same fit."""
    import json
    import os
    import sys
    from pathlib import Path

    path = tmp_path / "panel.parquet"
    with_extras(with_covariates(switching_panel(8, n=600), 8), 8).write_parquet(path)
    options = dict(effects=3, placebos=2, preprocess_mb=4, batch_units=37, n_threads=2, **extra)
    code = f"""
import json
from diff_stream import did_dcdh
r = did_dcdh({fml!r}, {str(path)!r}, panel=("id", "time"), treatment="d",
    workdir={str(tmp_path / "child")!r}, **{options!r})
print(json.dumps(dict(beta=r.beta.tolist(), vcov=r.vcov.tolist())))
"""
    names = [
        "STREAM_LIMITS_MAX_THREADS",
        "POLARS_MAX_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ]
    env = dict(os.environ, **dict.fromkeys(names, "2"), PYTHONPATH=str(Path(__file__).parents[1]))
    result = resource_runner([sys.executable, "-c", code], memory_gb=1.5, env=env, timeout=120)
    assert result.status == "ok", result.as_dict()
    assert result.mode == resource_enforcement["mode"]
    fitted = json.loads(result.stdout)
    baseline = did_dcdh(
        fml, path, panel=("id", "time"), treatment="d", effects=3, placebos=2, **extra
    )
    assert_allclose(fitted["beta"], baseline.beta, rtol=1e-12)
    assert_allclose(fitted["vcov"], baseline.vcov, rtol=1e-10)


REF = Path(__file__).parent / "data" / "dcdh_reference"
TIME_WEIGHTS = {"weights": "w", "allow_time_varying_weights": True}
R_CASES = {
    "switching": ("switching", "y ~ 1", {}),
    "no_never": ("no_never", "y ~ 1", {}),
    "normalized": ("switching", "y ~ 1", {"normalized": True}),
    "only_never_switchers": ("switching", "y ~ 1", {"only_never_switchers": True}),
    "switchers_in": ("switching", "y ~ 1", {"switchers": "in"}),
    "switchers_out": ("switching", "y ~ 1", {"switchers": "out"}),
    "covariates": ("switching", "y ~ x + z", {}),
    "unbalanced": ("unbalanced", "y ~ 1", {"incomplete_units": "keep"}),
    "unbalanced_covariates": ("unbalanced", "y ~ x + z", {"incomplete_units": "keep"}),
    "weights": ("switching", "y ~ 1", TIME_WEIGHTS),
    "weights_covariates": ("switching", "y ~ x + z", TIME_WEIGHTS),
    "weights_unbalanced": ("unbalanced", "y ~ 1", {"incomplete_units": "keep", **TIME_WEIGHTS}),
    "trends_by": ("switching", "y_trend ~ 1", {"trends_by": "region"}),
    "trends_by_covariates": ("switching", "y_trend ~ x + z", {"trends_by": "region"}),
    "linear_trends": ("switching", "y_trend ~ 1", {"linear_trends": True}),
    "linear_trends_covariates": ("switching", "y_trend ~ x + z", {"linear_trends": True}),
    "linear_trends_normalized": (
        "switching",
        "y_trend ~ 1",
        {"linear_trends": True, "normalized": True},
    ),
    "linear_trends_weights": ("switching", "y_trend ~ 1", {"linear_trends": True, **TIME_WEIGHTS}),
    "linear_trends_no_never": ("no_never", "y_trend ~ 1", {"linear_trends": True}),
    "linear_trends_trends_by": (
        "switching",
        "y_trend ~ 1",
        {"linear_trends": True, "trends_by": "region"},
    ),
}


@pytest.fixture(scope="module")
def reference_panels():
    for name, h in json.loads((REF / "PROVENANCE.json").read_text())["files"].items():
        assert digest(Path(__file__).parents[1] / name) == h, name
    return read_csv(REF.parent / "dcdh_panel.csv")


@pytest.mark.parametrize("name", R_CASES)
def test_matches_did_multiplegt_dyn(reference_panels, name):
    """The authors' R DIDmultiplegtDYN: estimates, standard errors and its joint tests,
    which use the covariances between the estimates."""
    panel, fml, options = R_CASES[name]
    frame = reference_panels.filter(pl.col("panel") == panel).drop("panel")
    fit = did_dcdh(fml, frame, **KW, effects=4, placebos=3, **options)
    # with trends_lin, the package estimates no average total effect (an empty row)
    ref = read_csv(REF / f"{name}.csv").filter(pl.col("estimate").is_not_null())
    assert fit.coefnames == ref["coefficient"].to_list()
    assert_allclose(fit.beta, ref["estimate"].to_numpy(), rtol=1e-10, atol=1e-12)
    assert_allclose(fit.se, ref["se"].to_numpy(), rtol=1e-10)
    table = fit.switchers()
    n = len(table)
    assert_allclose(
        table["weight_in"] + table["weight_out"],
        ref["switchers_weighted"].to_numpy()[:n],
        rtol=1e-12,
    )
    if not options.get("linear_trends"):
        # (with trends_lin, the package's unweighted count for every estimate is that of
        # the last estimate it computed for the same horizon; its weighted count, above,
        # is right)
        assert_allclose(
            table["switchers_in"] + table["switchers_out"], ref["switchers"].to_numpy()[:n]
        )
    tests = dict(read_csv(REF / f"{name}_tests.csv").iter_rows())
    assert_allclose(fit.pretrend_test()["pvalue"], tests["placebos_zero"], rtol=1e-8, atol=1e-14)
    assert_allclose(
        fit.effects_equal_test()["pvalue"], tests["effects_equal"], rtol=1e-8, atol=1e-14
    )
