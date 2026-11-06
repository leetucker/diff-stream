"""Native feTR references, independent projections and sensitivity optimization."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import minimize

from benchmarks._files import read_csv
from benchmarks.twfe_weights_validation import check, configs, dense, fit, provenance, sample
from diff_stream import TWFEWeightResult, twfe_weights

KW = dict(panel=("id", "time"), treatment="d", threads=2, preprocess_mb=4, batch_units=7)


@pytest.mark.parametrize("config", configs(), ids=lambda c: c["name"])
def test_native_r_and_independent_original_row_projection(config):
    check(config)


def test_frozen_reference_provenance():
    provenance()


@pytest.mark.parametrize("concentration", [1e4, 1e6])
def test_concentrated_weights_against_original_row_wls(concentration):
    data = (
        sample("unbalanced")
        .with_columns(
            pl.when(pl.col("id") == 4)
            .then(pl.col("w") * concentration)
            .otherwise(pl.col("w"))
            .alias("w")
        )
        .sort(["id", "time"])
    )
    oracle = dense(data, True, "w")
    with twfe_weights(
        "y~x+z",
        data,
        weights="w",
        allow_time_varying_weights=True,
        panel_mode="unbalanced",
        keep_cells=True,
        **KW,
    ) as result:
        cells = result.cells().collect().sort(["unit", "time"])
        assert_allclose(result.twfe_estimate, oracle["beta"], atol=2e-8, rtol=0)
        assert_allclose(cells["weight"], oracle["q"], atol=2e-8, rtol=0)
        json.loads(result.summary_json())


@pytest.mark.parametrize("seed", range(5))
def test_known_heterogeneous_effects_and_reversals(seed):
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(16), 7)
    t = np.tile(np.arange(1, 8), 16)
    d = rng.integers(0, 2, len(t))
    tau = rng.normal(1, 2, len(t))
    y = rng.normal(size=16)[ids] + np.sin(t) + tau * d
    data = pl.DataFrame(dict(id=ids, time=t, d=d, y=y, w=rng.uniform(0.2, 3, len(t))))
    data = data.filter(~((pl.col("id") % 3 == 0) & (pl.col("time") == 4)))
    with twfe_weights(
        "y~1",
        data,
        weights="w",
        allow_time_varying_weights=True,
        panel_mode="unbalanced",
        keep_cells=True,
        **KW,
    ) as r:
        c = r.cells().collect().sort(["unit", "time"])
        a = (c["unit"].to_numpy() * 7 + c["time"].to_numpy() - 1).astype(int)
        assert_allclose(c["weight"].to_numpy() @ tau[a], r.twfe_estimate, atol=2e-10, rtol=0)
        assert_allclose(c["weight"].sum(), 1, atol=2e-12, rtol=0)
        assert (c.filter(pl.col("treatment") == 0)["weight"] == 0).all()
        assert c.filter(pl.col("treatment") == 0)["relative_weight"].null_count() == len(
            c.filter(pl.col("treatment") == 0)
        )


@pytest.mark.parametrize("controlled", [False, True])
def test_sensitivity_against_constrained_quadratic_optimization(controlled):
    c = next(
        c
        for c in configs()
        if c["case"] == "absorbing" and c["controlled"] == controlled and c["weight"] == "w"
    )
    with fit(c, keep_cells=True) as r:
        cells = r.cells().collect().filter(pl.col("treatment") == 1)
        q, p = [cells[k].to_numpy() for k in ["weight", "treated_share"]]
        beta = r.twfe_estimate
        # Minimum population SD with p'tau=0 and q'tau=beta.
        v = np.sum(p * (q / p - 1) ** 2)
        tau = beta * (q / p - 1) / v
        assert_allclose(p @ tau, 0, atol=1e-10)
        assert_allclose(q @ tau, beta, atol=1e-10)
        first = r.sensitivity().row(0, named=True)["value"]
        assert_allclose(
            np.sqrt(p @ (tau * tau)) * np.sqrt((len(p) - 1) / len(p)), first, atol=1e-10
        )
        # Opposite-sign effects: tau=-sign(beta)*u, u>=0, q'u=-abs(beta).
        start = np.where(q < 0, abs(beta) / (-q[q < 0].sum()), 0)

        def objective(u):
            return np.sum(p * u * u) - (p @ u) ** 2

        def gradient(u):
            return 2 * p * (u - p @ u)

        solved = minimize(
            objective,
            start,
            jac=gradient,
            bounds=[(0, None)] * len(q),
            constraints=[dict(type="eq", fun=lambda u: q @ u + abs(beta), jac=lambda u: q)],
            method="SLSQP",
            options=dict(maxiter=1000, ftol=1e-12),
        )
        assert solved.success, solved.message
        assert_allclose(
            np.sqrt(solved.fun), r.sensitivity().row(1, named=True)["value"], atol=2e-6, rtol=0
        )


@pytest.mark.parametrize("scale", [1e-8, 1e8])
def test_scaling_aliases_input_paths_and_retained_export(scale, tmp_path):
    data = sample("absorbing")
    with twfe_weights(
        "y~x+z", data, weights="w", allow_time_varying_weights=True, keep_cells=True, **KW
    ) as a:
        raw = a.cells().collect().sort(["unit", "time"])
        changed = data.with_columns(
            pl.col("x") * scale,
            pl.col("z") / scale,
            pl.col("w") * scale,
            (pl.col("id") + 1000).cast(pl.String).alias("id"),
            (2 * pl.col("x") * scale).alias("alias"),
        ).sample(fraction=1, shuffle=True, seed=8)
        path = tmp_path / "input.parquet"
        changed.write_parquet(path)
        work = tmp_path / "work"
        with twfe_weights(
            "y~x+z+alias",
            path,
            weights="w",
            allow_time_varying_weights=True,
            keep_cells=True,
            workdir=work,
            **{**KW, "batch_units": 3, "preprocess_mb": 1},
        ) as b:
            assert isinstance(b, TWFEWeightResult)
            assert_allclose(a.twfe_estimate, b.twfe_estimate, atol=2e-9, rtol=0)
            out = (
                b.cells()
                .collect()
                .with_columns((pl.col("unit").cast(pl.Int64) - 1000).alias("unit"))
                .sort(["unit", "time"])
            )
            assert_allclose(raw["weight"], out["weight"], atol=2e-9, rtol=0)
            assert "alias" in b.diagnostics["absorbed_or_redundant_columns"]
            b.write_cells(tmp_path / "export.parquet")
            saved = b.summary_json(tmp_path / "summary.json")
            assert json.loads(saved)["weights"]["raw_weight_sum"] == pytest.approx(1)
            assert sum(len(batch) for batch in b.iter_cells(batch_rows=13)) == len(data)
            with pytest.raises(FileExistsError):
                b.write_cells(tmp_path / "export.parquet")
        assert not list(work.iterdir())
        assert pl.read_parquet(tmp_path / "export.parquet").height == len(data)
        assert b.summary().height == 3
        with pytest.raises(ValueError, match="cell storage"):
            b.cells()


def test_cohort_convenience_and_sample_policies(tmp_path):
    data = sample("absorbing").with_columns(
        read_csv(Path(__file__).parent / "data/bacon_extended_panel.csv")["first_treat"]
    )
    # Derive treatment directly from positive pre-sample/future onsets too.
    data = data.with_columns(
        pl.when(pl.col("first_treat") == 2)
        .then(1)
        .when(pl.col("first_treat") == 0)
        .then(100)
        .otherwise(pl.col("first_treat"))
        .alias("first_treat")
    )
    data = data.with_columns(
        ((pl.col("first_treat") > 0) & (pl.col("time") >= pl.col("first_treat")))
        .cast(pl.Float64)
        .alias("d")
    )
    a = twfe_weights("y~x+z", data, **KW)
    b = twfe_weights(
        "y~x+z", data, cohort="first_treat", **{k: v for k, v in KW.items() if k != "treatment"}
    )
    assert_allclose(a.twfe_estimate, b.twfe_estimate, atol=1e-12, rtol=0)
    b.cleanup()
    with pytest.raises(ValueError, match="exactly one"):
        twfe_weights("y~1", data, cohort="first_treat", **KW)
    unbalanced = sample("unbalanced")
    with pytest.raises(ValueError, match="balanced panels"):
        twfe_weights("y~1", unbalanced, workdir=tmp_path, **KW)
    r = twfe_weights("y~1", unbalanced, panel_mode="complete", workdir=tmp_path, **KW)
    assert r.diagnostics["dropped_units"] == 6 and r.diagnostics["dropped_rows"] > 0
    assert not list(tmp_path.iterdir())


def test_zero_weights_zero_outcome_and_degenerate_sensitivity():
    constant = twfe_weights("y~1", sample("constant_multiplier"), **KW)
    assert constant.sensitivity().row(0, named=True)["status"] == "infinite_constant_multiplier"
    assert constant.sensitivity().row(1, named=True)["status"] == "infeasible_no_negative_weights"
    zero = twfe_weights("y~1", sample("zero_outcome"), **KW)
    assert zero.twfe_estimate == 0
    assert zero.sensitivity().row(0, named=True)["value"] == 0
    assert zero.sensitivity().row(1, named=True)["status"] == "undefined_zero_coefficient"
    coarse = twfe_weights("y~1", sample("absorbing"), zero_tol=1, **KW)
    assert coarse.summary().filter(pl.col("sign") == "zero")["n_cells"][0] > 0
    assert coarse.summary_dict()["weights"]["raw_weight_sum"] == pytest.approx(1)


@pytest.mark.parametrize(
    "case",
    [
        "all_zero",
        "all_one",
        "group_constant",
        "explained",
        "duplicate",
        "missing",
        "bad_weight",
        "changing_weight",
        "continuous",
    ],
)
def test_identification_and_source_validation(case, tmp_path):
    data = sample("absorbing")
    formula = "y~1"
    if case == "all_zero":
        data = data.with_columns(pl.lit(0).alias("d"))
    if case == "all_one":
        data = data.with_columns(pl.lit(1).alias("d"))
    if case == "group_constant":
        data = data.with_columns((pl.col("id") % 2).alias("d"))
    if case == "explained":
        data = data.with_columns(pl.col("d").alias("perfect"))
        formula = "y~perfect"
    if case == "duplicate":
        data = pl.concat([data, data.head(1)])
    if case == "missing":
        data = data.with_columns(
            pl.when(pl.col("id") == 0).then(None).otherwise(pl.col("y")).alias("y")
        )
    if case == "bad_weight":
        data = data.with_columns(pl.lit(0.0).alias("unit_w"))
    if case == "continuous":
        data = data.with_columns((0.5 * pl.col("d")).alias("d"))
    weight = "w" if case == "changing_weight" else "unit_w"
    with pytest.raises(ValueError):
        twfe_weights(formula, data, weights=weight, workdir=tmp_path, **KW)
    assert not list(tmp_path.iterdir())


def test_resource_admission_profile_and_no_retained_cells(tmp_path):
    data = sample("absorbing")
    with pytest.raises(MemoryError):
        twfe_weights("y~x", data, memory_gb=0.01, workdir=tmp_path, **KW)
    with pytest.raises(ValueError, match="max_design_columns"):
        twfe_weights("y~x", data, max_design_columns=2, workdir=tmp_path, **KW)
    profile = twfe_weights("y~x", data, profile=True, workdir=tmp_path, **KW)
    assert profile.notes["type"] == "feTR"
    r = twfe_weights("y~1", data, workdir=tmp_path, **KW)
    with pytest.raises(ValueError, match="keep_cells=True"):
        r.cells()
    assert not list(tmp_path.iterdir())


@pytest.mark.resource
def test_twfe_weights_constrained(resource_runner, resource_enforcement, tmp_path):
    path = Path(__file__).parent / "data/bacon_extended_panel.csv"
    work = tmp_path / "child"
    code = f"""from diff_stream import twfe_weights
import polars as pl
import pyarrow.parquet as pq
d=pl.read_csv({str(path)!r})
d=d.join(pl.DataFrame({{'rep':range(200)}}),how='cross').with_columns((pl.col('id')+100*pl.col('rep')).alias('id')).drop('rep')
with twfe_weights('y~x+z',d,panel=('id','time'),cohort='first_treat',weights='w',threads=2,keep_cells=True,batch_units=7,preprocess_mb=4,workdir={str(work)!r}) as r:
 with pq.ParquetFile(r._path,pre_buffer=False) as pf:
  assert pf.metadata.num_row_groups < 8
 r.write_cells({str(tmp_path / "cells.parquet")!r})
 print(r.summary_json())
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
    assert json.loads(result.stdout)["weights"]["raw_weight_sum"] == pytest.approx(1)
    expected = read_csv(
        Path(__file__).parent / "data/bacon_extended_reference/models.csv", null_values="NA"
    ).filter(pl.col("name") == "plain_1_irrational")["twfe"][0]
    assert_allclose(json.loads(result.stdout)["twfe_estimate"], expected, atol=2e-9, rtol=0)
    if result.kernel:
        assert result.kernel["events"]["oom_kill"] == 0
    assert not list(work.iterdir())
