"""Balanced covariate/always-treated Bacon: native R and independent WLS."""

import json
import os
import sys

import polars as pl
import pytest
from numpy.testing import assert_allclose

from benchmarks.bacon_extended_validation import (
    CASES,
    check,
    dense_oracle,
    provenance,
    retained,
    sample,
)
from diff_stream import bacon_decomp, cs_stream

KW = dict(panel=("id", "time"), cohort="first_treat", threads=2, preprocess_mb=4, batch_units=7)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize(
    "kind,weight", [("equal", None), ("integer", "unit_w"), ("irrational", "w")]
)
def test_native_bacon_and_independent_row_projection(case, controlled, kind, weight):
    fit = bacon_decomp(
        "y~x+z" if controlled else "y~1",
        sample(case),
        panel_mode="complete" if case == "complete" else "balanced",
        weights=weight,
        **KW,
    )
    check(fit, case, controlled, kind)
    twfe, omega, within, pairs = dense_oracle(retained(sample(case), case), controlled, weight)
    assert_allclose(fit.twfe_estimate, twfe, atol=2e-9, rtol=0)
    assert_allclose(fit.summary()["weight"].sum(), 1, atol=2e-12, rtol=0)
    assert_allclose(fit.summary()["contribution"].sum(), fit.twfe_estimate, atol=2e-9, rtol=0)
    assert abs(fit.reconstruction_error) < 2e-9
    json.loads(fit.summary_json(comparisons=True))
    if controlled:
        assert_allclose(fit.within_weight, omega, atol=2e-9, rtol=0)
        assert_allclose(fit.within_contribution, within, atol=2e-9, rtol=0)
        table = fit.comparisons()
        assert_allclose(table["weight"].sum(), 1 - omega, atol=2e-12, rtol=0)
        assert_allclose(table["between_weight"].sum(), 1, atol=2e-12, rtol=0)
        assert table["pre_start"].null_count() == len(table)
        for row in table.to_dicts():
            w, b = pairs[tuple(sorted((row["treated_cohort"], row["control_cohort"])))]
            assert_allclose(row["weight"], w, atol=2e-12, rtol=0)
            if row["weight"] > 1e-14:
                assert_allclose(row["estimate"], b, atol=2e-8, rtol=0)
    else:
        with pytest.raises(ValueError, match="covariate-adjusted"):
            fit.within_timing()
        if "always" in case:
            assert "later_vs_always" in fit.comparisons()["comparison_type"]


def test_reference_provenance():
    provenance()


@pytest.mark.parametrize("weight", [None, "unit_w"])
def test_absorbed_and_redundant_covariates(weight):
    d = sample("plain").with_columns(
        pl.col("x") * 2, "id", pl.col("time").cast(pl.Float64).alias("calendar")
    )
    baseline = bacon_decomp("y~1", d, weights=weight, **KW)
    absorbed = bacon_decomp("y~id+calendar", d, weights=weight, **KW)
    assert_allclose(absorbed.twfe_estimate, baseline.twfe_estimate, atol=2e-10, rtol=0)
    assert absorbed.within_weight == 0
    assert absorbed.within_estimate is None
    assert absorbed.within_contribution == 0
    x = d.with_columns((2 * pl.col("x")).alias("duplicate"))
    a = bacon_decomp("y~x+z", x, weights=weight, **KW)
    b = bacon_decomp("y~x+z+duplicate", x, weights=weight, **KW)
    assert_allclose(a.twfe_estimate, b.twfe_estimate, atol=2e-10, rtol=0)
    assert_allclose(a.within_weight, b.within_weight, atol=2e-10, rtol=0)
    assert "duplicate" in b.diagnostics["absorbed_or_redundant_columns"]


@pytest.mark.parametrize("scale", [1e-8, 1e8])
def test_covariate_and_weight_units_partition_and_cleanup(scale, tmp_path):
    d = sample("always")
    a = bacon_decomp("y~x+z", d, weights="w", **KW)
    changed = d.with_columns(
        pl.col("x") * scale,
        pl.col("z") / scale,
        pl.col("w") * scale,
        ("unit_" + pl.col("id").cast(pl.String)).alias("id"),
    ).sample(fraction=1, shuffle=True, seed=9)
    path = tmp_path / "input.parquet"
    changed.write_parquet(path)
    work = tmp_path / "scratch"
    work.mkdir()
    b = bacon_decomp(
        "y~x+z", path, weights="w", workdir=work, **{**KW, "preprocess_mb": 1, "batch_units": 3}
    )
    assert_allclose(a.twfe_estimate, b.twfe_estimate, atol=2e-9, rtol=0)
    assert_allclose(a.comparisons()["weight"], b.comparisons()["weight"], atol=2e-9, rtol=0)
    assert_allclose(a.within_weight, b.within_weight, atol=2e-9, rtol=0)
    assert not list(work.iterdir())


def test_guards_identification_profile_and_estimator_boundary(tmp_path):
    d = sample("always")
    with pytest.raises(ValueError, match="clean pre-period"):
        cs_stream("y~1", d, panel=("id", "time"), cohort="first_treat")
    explained = d.with_columns(
        ((pl.col("first_treat") > 0) & (pl.col("time") >= pl.col("first_treat")))
        .cast(pl.Float64)
        .alias("perfect")
    )
    with pytest.raises(ValueError, match="collinear"):
        bacon_decomp("y~perfect", explained, workdir=tmp_path, **KW)
    with pytest.raises(ValueError, match="unbalanced"):
        bacon_decomp("y~x", d, panel_mode="unbalanced", **KW)
    with pytest.raises(ValueError, match="batch_units"):
        bacon_decomp("y~x", d, **{**KW, "batch_units": 0})
    with pytest.raises(MemoryError):
        bacon_decomp("y~x", d, memory_gb=0.01, workdir=tmp_path, **KW)
    prof = bacon_decomp("y~x+z", d, profile=True, workdir=tmp_path, **KW)
    assert prof.notes["covariate_adjusted"] and prof.notes["always_treated_units"] > 0
    assert not list(tmp_path.iterdir())


@pytest.mark.resource
def test_extended_bacon_constrained(resource_runner, resource_enforcement, tmp_path):
    from benchmarks.bacon_extended_validation import DATA

    work = tmp_path / "child"
    code = f"""import json
from diff_stream import bacon_decomp
import polars as pl
x=pl.read_csv({str(DATA)!r}).with_columns(pl.when(pl.col('first_treat')==2).then(1).otherwise(pl.col('first_treat')).alias('first_treat'))
r=bacon_decomp('y~x+z',x,panel=('id','time'),cohort='first_treat',weights='w',threads=2,batch_units=7,preprocess_mb=4,workdir={str(work)!r})
print(r.summary_json(comparisons=True))
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
    got = json.loads(result.stdout)
    baseline = bacon_decomp("y~x+z", sample("always"), weights="w", **KW)
    assert_allclose(got["twfe_estimate"], baseline.twfe_estimate, atol=2e-10, rtol=0)
    assert_allclose(got["within_timing"]["weight"], baseline.within_weight, atol=2e-10, rtol=0)
    if result.kernel:
        assert result.kernel["events"]["oom_kill"] == 0
    assert not list(work.iterdir())
