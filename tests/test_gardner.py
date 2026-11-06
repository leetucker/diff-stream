"""Authors' reference and independent stacked-equation Gardner validation."""

import json
import os
import sys

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from benchmarks._estimators import public
from benchmarks._files import read_csv
from benchmarks.gardner_validation import (
    CASES,
    DATA,
    REF,
    configuration,
    dense_oracle,
    verify_provenance,
)
from diff_stream import TwoStageResult, did_two_stage

KW = dict(panel=("id", "time"), cohort="first_treat", n_threads=2, preprocess_mb=4, batch_units=37)


def reference_check(fit, prefix):
    ref = read_csv(REF / f"{prefix}.csv")
    names = ref["term"].to_list()
    ix = [names.index(n) for n in fit.coefnames]
    V = read_csv(REF / f"{prefix}_vcov.csv", has_header=False).to_numpy()
    assert_allclose(fit.beta, ref["estimate"].to_numpy()[ix], atol=2e-9, rtol=0)
    assert_allclose(fit.vcov, V[np.ix_(ix, ix)], atol=2e-9, rtol=0)


def test_frozen_reference_provenance():
    manifest = verify_provenance()
    assert "Unmodified" in manifest["execution"]


@pytest.mark.parametrize("panel_mode", ["balanced", "unbalanced"])
@pytest.mark.parametrize("anticipation", [0, 1])
@pytest.mark.parametrize("specification", ["static", "dynamic"])
@pytest.mark.parametrize("case", CASES)
def test_authors_and_dense_joint_covariance(panel_mode, anticipation, specification, case):
    d = read_csv(
        DATA / ("richer_panel.csv" if panel_mode == "balanced" else "richer_unbalanced_panel.csv")
    )
    formula, weight = configuration(case)
    fit = did_two_stage(
        formula,
        d,
        **KW,
        incomplete_units=public("incomplete_units", panel_mode),
        anticipation=anticipation,
        event_study=public("event_study", specification),
        weights=weight,
        allow_time_varying_weights=True,
    )
    assert isinstance(fit, TwoStageResult)
    reference_check(fit, f"{panel_mode}_{anticipation}_{case}_{specification}")
    names, beta, V = dense_oracle(d, case, specification, anticipation)
    assert fit.coefnames == names
    assert_allclose(fit.beta, beta, atol=2e-9, rtol=0)
    assert_allclose(fit.vcov, V, atol=2e-9, rtol=0)
    assert fit.n_obs == len(d)
    assert fit.n_units == d["id"].n_unique()
    assert len(fit.tidy()) == len(beta)
    assert np.linalg.eigvalsh(fit.vcov).min() > -1e-10


@pytest.mark.parametrize("panel_mode", ["balanced", "unbalanced"])
@pytest.mark.parametrize("anticipation", [0, 1])
def test_custom_reference_and_postfit_selection(panel_mode, anticipation):
    d = read_csv(
        DATA / ("richer_panel.csv" if panel_mode == "balanced" else "richer_unbalanced_panel.csv")
    )
    formula, weight = configuration("extra_fe_varying")
    fit = did_two_stage(
        formula,
        d,
        **KW,
        incomplete_units=public("incomplete_units", panel_mode),
        anticipation=anticipation,
        weights=weight,
        allow_time_varying_weights=True,
        reference_period=-anticipation - 2,
    )
    reference_check(fit, f"{panel_mode}_{anticipation}_custom_reference")
    _, beta, V = dense_oracle(d, "extra_fe_varying", "dynamic", anticipation, -anticipation - 2)
    assert_allclose(fit.beta, beta, atol=2e-9, rtol=0)
    assert_allclose(fit.vcov, V, atol=2e-9, rtol=0)
    selected = fit.select([0, 1])
    ix = [fit.coefnames.index(f"event::{h}") for h in [0, 1]]
    assert_allclose(selected.beta, fit.beta[ix])
    assert_allclose(selected.vcov, fit.vcov[np.ix_(ix, ix)])
    selected.beta[:] = 999
    assert not np.any(fit.beta == 999)
    assert set(fit.support()["aggregate"]) == set(fit.coefnames)
    assert fit.pretrend_test(horizons=[-anticipation - 3])["n_restrictions"] == 1
    assert json.loads(fit.summary_json())["depvar"] == "y"


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("specification", ["static", "dynamic"])
def test_castle_public_application(weighted, specification):
    from benchmarks.castle_replication import prepare_data
    from benchmarks.castle_replication import verify_provenance as verify_castle

    verify_castle()
    d = prepare_data(shift=1).rename({"state": "id", "year": "time", "l_homicide": "y"})
    fit = did_two_stage(
        "y ~ 1",
        d,
        **KW,
        event_study=public("event_study", specification),
        weights="popwt" if weighted else None,
        allow_time_varying_weights=True,
    )
    reference_check(fit, f"castle_{str(weighted).upper()}_{specification}")
    oracle = d.with_columns(pl.col("popwt").alias("w"))
    names, beta, V = dense_oracle(
        oracle, "plain_varying" if weighted else "plain_equal", specification
    )
    assert fit.coefnames == names
    assert_allclose(fit.beta, beta, atol=2e-9, rtol=0)
    assert_allclose(fit.vcov, V, atol=2e-9, rtol=0)


def test_dispatch_crv_weights_scaling_and_complete_sample(tmp_path):
    d = read_csv(DATA / "richer_unbalanced_panel.csv")
    kw = dict(
        **KW,
        incomplete_units="drop",
        weights="w",
        allow_time_varying_weights=True,
        workdir=tmp_path,
    )
    fit = did_two_stage("y ~ x", d, event_study=False, **kw)
    scaled = did_two_stage(
        "y ~ x", d.with_columns(pl.col("w") * 1e10), event_study=False, vcov="CRV1", **kw
    )
    n = fit.n_units
    assert_allclose(fit.beta, scaled.beta, atol=1e-11, rtol=0)
    assert_allclose(fit.vcov * n / (n - 1), scaled.vcov, atol=1e-11, rtol=0)
    assert_allclose(fit.with_vcov("CRV1").vcov, scaled.vcov, atol=1e-11, rtol=0)
    assert_allclose(scaled.with_vcov("CRV0").vcov, fit.vcov, atol=1e-11, rtol=0)
    assert fit.n_units < d["id"].n_unique()
    assert fit.diagnostics["dropped_units"] > 0
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError, match="event_study=True"):
        fit.select([0])
    with pytest.raises(ValueError, match="event-time"):
        fit.pretrend_test()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"event_study": "bad"},
        {"reference_period": 0},
        {"reference_period": True},
        {"anticipation": -1},
        {"anticipation": True},
        {"batch_units": 0},
        {"n_threads": 0},
        {"preprocess_mb": 0},
        {"incomplete_units": "bad"},
        {"vcov": "iid"},
        {"allow_time_varying_weights": 1},
        {"event_study": False, "reference_period": -1},
    ],
)
def test_invalid_options_fail_before_staging(kwargs, tmp_path):
    with pytest.raises(ValueError):
        did_two_stage(
            "y ~ 1",
            "missing.parquet",
            panel=("id", "time"),
            cohort="first_treat",
            workdir=tmp_path,
            **kwargs,
        )
    assert not list(tmp_path.iterdir())


def test_identification_errors_cleanup_and_profile(tmp_path):
    d = read_csv(DATA / "richer_panel.csv")
    kw = dict(**KW, workdir=tmp_path)
    with pytest.raises(ValueError, match="reference_period"):
        did_two_stage("y ~ 1", d, reference_period=-100, **kw)
    # A column with no clean-sample variation cannot be extrapolated to treatment.
    bad = d.with_columns(
        ((pl.col("first_treat") != 0) & (pl.col("time") >= pl.col("first_treat")))
        .cast(pl.Float64)
        .alias("bad")
    )
    with pytest.raises(ValueError, match="Gardner untreated"):
        did_two_stage("y ~ bad", bad, **kw)
    with pytest.raises(ValueError, match="constant within unit"):
        did_two_stage("y ~ 1", d, weights="w", **kw)
    with pytest.raises(MemoryError):
        did_two_stage("y ~ 1", d, max_memory_gb=0.01, **kw)
    with pytest.raises(TypeError, match="pretrend_leads"):
        did_two_stage("y ~ 1", d, pretrend_leads=1, **kw)
    profile = did_two_stage("y ~ x", d, profile=True, **kw)
    assert profile.model == "diff-stream did_two_stage"
    assert not list(tmp_path.iterdir())


def test_shuffled_partition_batches_and_missing_unit_history(tmp_path):
    d = read_csv(DATA / "richer_unbalanced_panel.csv")
    kw = dict(
        panel=("id", "time"),
        cohort="first_treat",
        n_threads=2,
        incomplete_units="keep",
        weights="w",
        allow_time_varying_weights=True,
        preprocess_mb=1,
        workdir=tmp_path,
    )
    base = did_two_stage("y ~ x + I(x**2) | region^time", d, batch_units=512, **kw)
    path = tmp_path / "input.parquet"
    d.with_columns(
        ("unit_" + pl.col("id").cast(pl.String).str.pad_start(128, "0")).alias("id")
    ).sample(fraction=1, shuffle=True, seed=1).write_parquet(path, row_group_size=len(d))
    fit = did_two_stage("y ~ x + I(x**2) | region^time", path, batch_units=17, **kw)
    assert_allclose(fit.beta, base.beta, atol=1e-10, rtol=0)
    assert_allclose(fit.vcov, base.vcov, atol=1e-10, rtol=0)
    target = d.filter(pl.col("first_treat") > 0)["id"][0]
    bad = d.filter((pl.col("id") != target) | (pl.col("time") >= pl.col("first_treat")))
    with pytest.raises(ValueError, match="positive-weight|untreated observation"):
        did_two_stage("y ~ 1", bad, batch_units=31, **kw)
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.resource
def test_gardner_constrained_fit(resource_runner, resource_enforcement, tmp_path):
    source = DATA / "richer_unbalanced_panel.csv"
    work = tmp_path / "child"
    code = f"""import json,polars as pl
from diff_stream import did_two_stage
r=did_two_stage('y ~ x + I(x**2) | region^time',pl.read_csv({str(source)!r}),panel=('id','time'),cohort='first_treat',incomplete_units='keep',weights='w',allow_time_varying_weights=True,n_threads=2,preprocess_mb=4,batch_units=37,workdir={str(work)!r})
print(json.dumps(dict(names=r.coefnames,beta=r.beta.tolist(),vcov=r.vcov.tolist())))
"""
    env = dict(
        os.environ,
        STREAM_LIMITS_MAX_THREADS="2",
        POLARS_MAX_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        MPLCONFIGDIR=str(tmp_path / "mpl"),
    )
    result = resource_runner([sys.executable, "-c", code], memory_gb=1.5, timeout=90, env=env)
    assert result.status == "ok", result.stderr
    assert result.mode == resource_enforcement["mode"]
    if result.kernel:
        assert result.kernel["events"]["oom_kill"] == 0
    fit = did_two_stage(
        "y ~ x + I(x**2) | region^time",
        read_csv(source),
        **KW,
        incomplete_units="keep",
        weights="w",
        allow_time_varying_weights=True,
    )
    value = json.loads(result.stdout)
    assert value["names"] == fit.coefnames
    assert_allclose(value["beta"], fit.beta, atol=1e-10, rtol=0)
    assert_allclose(value["vcov"], fit.vcov, atol=1e-10, rtol=0)
    assert not list(work.iterdir())


@pytest.mark.parametrize("scale", [1e-8, 1e8])
def test_covariate_units_do_not_change_fit(scale):
    d = read_csv(DATA / "richer_unbalanced_panel.csv")
    kw = dict(**KW, incomplete_units="keep", weights="w", allow_time_varying_weights=True)
    base = did_two_stage("y ~ x", d, **kw)
    scaled = did_two_stage("y ~ x", d.with_columns(pl.col("x") * scale), **kw)
    assert_allclose(scaled.beta, base.beta, atol=1e-10, rtol=0)
    assert_allclose(scaled.vcov, base.vcov, atol=1e-10, rtol=0)
