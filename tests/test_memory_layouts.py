"""Source projection and whole-unit partition invariants for difficult layouts."""

import json
import os

import polars as pl
import pytest
from numpy.testing import assert_allclose

from benchmarks._estimators import ESTIMATORS, fit_estimator, public
from benchmarks._files import read_csv
from diff_stream.simulate import simulate_conditional_panel, simulate_panel


@pytest.mark.parametrize("estimator", ["cs", "cs_conditional", "sunab", "bjs"])
@pytest.mark.parametrize("unbalanced", [False, True])
def test_long_ids_shuffled_single_row_group_and_unused_invalid_values(
    estimator, unbalanced, tmp_path
):
    conditional = estimator == "cs_conditional"
    data = (simulate_conditional_panel if conditional else simulate_panel)(5000, 5, seed=711)
    formula = "y ~ x + z + C(category) + x:C(category)" if conditional else "y ~ 1"
    if unbalanced:
        data = data.filter(~((pl.col("id") % 4 == 0) & (pl.col("time") == 3)))
    kw = dict(
        panel=("id", "time"),
        cohort="first_treat",
        estimator="cs" if conditional else estimator,
        n_threads=2,
        incomplete_units=public("incomplete_units", "unbalanced" if unbalanced else "balanced"),
    )
    # Non-finite unused values must not participate in validation or decoded
    # model width. Long labels and row shuffling force whole-unit repartitioning.
    source = data.with_columns(
        ("unit_" + pl.col("id").cast(pl.String).str.pad_start(192, "0")).alias("id"),
        *[pl.lit(float("nan")).alias(f"unused{j}") for j in range(64)],
    ).sample(fraction=1, shuffle=True, seed=14)
    path = tmp_path / "source.parquet"
    source.write_parquet(path, row_group_size=len(source))
    work = tmp_path / "scratch"
    work.mkdir()
    baseline = fit_estimator(formula, data, **kw)
    actual = None
    try:
        actual = fit_estimator(
            formula, str(path), preprocess_mb=1, batch_units=31, workdir=work, **kw
        )
        assert actual.n_units == baseline.n_units == 5000
        assert actual.n_obs == len(data)
        assert actual.diagnostics["partitions"] > 1
        assert actual.diagnostics["max_decoded_leaf_bytes"] <= 250000
        assert_allclose(actual.beta, baseline.beta, atol=1e-10, rtol=0)
        assert_allclose(actual.vcov, baseline.vcov, atol=1e-10, rtol=0)
        assert_allclose(
            actual.aggregate("overall").vcov,
            baseline.aggregate("overall").vcov,
            atol=1e-10,
            rtol=0,
        )
    finally:
        baseline.cleanup()
        if actual is not None:
            actual.cleanup()
    assert not list(work.iterdir())


def test_resource_generator_layout_changes_preserve_sample(tmp_path):
    from benchmarks import resource_checks as h

    spec = h.dataset(4300, width=17, pattern="entry_exit")
    left, right = tmp_path / "left.parquet", tmp_path / "right.parquet"
    h.generate(spec, left)
    variant = dict(
        spec, strings=True, id_padding=96, shuffled=True, unused_columns=8, row_group=65536
    )
    h.generate(variant, right)
    a = pl.read_parquet(left).sort("id", "time")
    b = (
        pl.read_parquet(right)
        .with_columns(pl.col("id").str.strip_prefix("unit_").cast(pl.Int64))
        .select(a.columns)
        .sort("id", "time")
    )
    assert a.equals(b)
    manifest = json.loads(right.with_suffix(".json").read_text())
    assert manifest["max_row_group_rows"] == len(a)


@pytest.mark.resource
@pytest.mark.kernel_memory
def test_real_cgroup_kernel_enforcement(verified_memory_parent, memory_runner):
    import sys

    result = memory_runner(
        [sys.executable, "-c", "x=bytearray(16*1024*1024); print('ok')"],
        mode="hard",
        memory_gb=128 * 2**20 / 1e9,
        cgroup_parent=verified_memory_parent,
    )
    assert result.status == "ok" and result.stdout.strip() == "ok"
    assert result.kernel["memory_max_bytes"] == 128 * 2**20
    assert result.kernel["events"]["oom_kill"] == 0


@pytest.mark.resource
@pytest.mark.parametrize("estimator", ["cs", "sunab", "bjs"])
def test_constrained_fit_matches_unconstrained_fit(
    resource_runner, resource_enforcement, estimator, tmp_path
):
    import sys
    from pathlib import Path

    import numpy as np

    root = Path(__file__).resolve().parents[1]
    formula = "y ~ x+z" if estimator == "cs" else "y ~ x + I(x**2) | region^time"
    extra = {"cs": {"effective_sample_warning": None}, "bjs": {"pretrend_leads": 2}}.get(
        estimator, {}
    )
    function = ESTIMATORS[estimator].__name__
    code = f"""
import json,polars as pl
from diff_stream import {function}

r={function}({formula!r},pl.read_csv({str(root / "tests/data/richer_unbalanced_panel.csv")!r}),
 panel=('id','time'),cohort='first_treat',incomplete_units='keep',
 weights='w',allow_time_varying_weights=True,n_threads=2,preprocess_mb=4,batch_units=37,
 workdir={str(tmp_path / "child")!r},**{extra!r})
try:
 a=r.aggregate('overall')
 values=dict(names=r.coefnames,beta=r.beta.tolist(),vcov=r.vcov.tolist(),att=a.beta.tolist())
 if {estimator == "bjs"!r}: values.update(placebo=r.placebos().beta.tolist(),placebo_vcov=r.placebos().vcov.tolist())
 print(json.dumps(values))
finally:r.cleanup()
"""
    env = dict(
        os.environ,
        STREAM_LIMITS_MAX_THREADS="2",
        POLARS_MAX_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        OMP_NUM_THREADS="2",
        NUMBA_NUM_THREADS="2",
        MPLCONFIGDIR=str(tmp_path / "mpl"),
    )
    result = resource_runner(
        [sys.executable, "-c", code],
        memory_gb=1.5,
        env=env,
        timeout=60,
    )
    assert result.status == "ok", result.as_dict()
    assert result.mode == resource_enforcement["mode"]
    if result.mode == "hard":
        assert result.kernel["events"]["oom_kill"] == 0
    else:
        assert result.kernel is None
    fitted = json.loads(result.stdout)
    # This is a resource test: compare with an unconstrained fit on the SAME
    # platform. The regular numerical suite owns the independent R references.
    baseline = fit_estimator(
        formula,
        read_csv(root / "tests/data/richer_unbalanced_panel.csv"),
        panel=("id", "time"),
        cohort="first_treat",
        estimator=estimator,
        incomplete_units="keep",
        weights="w",
        allow_time_varying_weights=True,
        n_threads=2,
        preprocess_mb=4,
        batch_units=37,
        workdir=tmp_path / "baseline",
        **extra,
    )
    try:
        assert fitted["names"] == baseline.coefnames
        np.testing.assert_allclose(fitted["beta"], baseline.beta, atol=1e-9, rtol=0)
        np.testing.assert_allclose(fitted["vcov"], baseline.vcov, atol=1e-9, rtol=0)
        np.testing.assert_allclose(
            fitted["att"], baseline.aggregate("overall").beta, atol=1e-9, rtol=0
        )
        if estimator == "bjs":
            np.testing.assert_allclose(
                fitted["placebo"], baseline.placebos().beta, atol=1e-9, rtol=0
            )
            np.testing.assert_allclose(
                fitted["placebo_vcov"], baseline.placebos().vcov, atol=1e-9, rtol=0
            )
    finally:
        baseline.cleanup()
    assert not list((tmp_path / "baseline").iterdir())
    assert not list((tmp_path / "child").iterdir())
