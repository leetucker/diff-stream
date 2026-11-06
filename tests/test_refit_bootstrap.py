"""Ordinary unit-pairs draws, full refits, fixed references and failure gates."""

import hashlib
import json
import os
import sys

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose
from scipy.stats import norm
from stream_limits import MemoryBudget

from benchmarks._estimators import ESTIMATORS, fit_estimator, public
from diff_stream import (
    BootstrapFailureWarning,
    RefitBootstrapResult,
    did_bjs,
    did_callaway_santanna,
    refit_bootstrap,
)
from diff_stream._core.resample import DiskGroupResampler
from diff_stream.simulate import simulate_conditional_panel

KW = dict(panel=("id", "time"), cohort="first_treat", n_threads=2, preprocess_mb=4, batch_units=37)


def small_data(last=False, unbalanced=False):
    data = simulate_conditional_panel(80, 5, seed=17).with_columns(
        ((0.3 + pl.col("id") % 4) * (1 + 0.05 * pl.col("time"))).alias("w")
    )
    if last:
        data = data.filter(pl.col("first_treat") != 0)
    if unbalanced:
        data = data.filter(~((pl.col("id") % 4 == 0) & (pl.col("time") == 2)))
    return data


@pytest.mark.parametrize("estimator", ["cs", "sunab", "bjs", "gardner", "etwfe"])
@pytest.mark.parametrize("last", [False, True])
@pytest.mark.parametrize("anticipation", [0, 1])
def test_every_draw_matches_explicit_unit_duplication(estimator, last, anticipation, tmp_path):
    data = small_data(last, True)
    options = dict(
        weights="w",
        allow_time_varying_weights=True,
        incomplete_units="keep",
        anticipation=anticipation,
    )
    if estimator == "cs":
        options["effective_sample_warning"] = None
    if estimator == "gardner":
        options["event_study"] = False
    target = None if estimator == "gardner" else "event_time"
    r = refit_bootstrap(
        "y~x",
        data,
        estimator=ESTIMATORS[estimator],
        reps=4,
        seed=29,
        aggregate=target,
        workdir=tmp_path,
        **KW,
        **options,
    )
    assert not r.failures().height
    selected = data.filter(pl.col("time") < 5 - anticipation) if last else data
    units = sorted(selected["id"].unique().to_list())
    rng = np.random.default_rng(29)
    for j in range(4):
        indices = rng.integers(0, len(units), size=len(units), dtype=np.int64)
        expected_hash = hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest()
        assert r.diagnostics["draw_plan_sha256"][j] == expected_hash
        rows = pl.concat(
            [
                selected.filter(pl.col("id") == units[int(u)]).with_columns(pl.lit(i).alias("boot"))
                for i, u in enumerate(indices)
            ]
        )
        with fit_estimator(
            "y~x",
            rows,
            panel=("boot", "time"),
            cohort="first_treat",
            estimator=estimator,
            last_treated_controls="never",
            n_threads=2,
            preprocess_mb=4,
            batch_units=37,
            **options,
        ) as fitted:
            value = fitted.aggregate(public("kind", target)) if target else fitted
            ix = [value.coefnames.index(n) for n in r.coefnames]
            assert_allclose(r._draws[j], value.beta[ix], atol=2e-9, rtol=0)
    assert_allclose(r.se, np.std(r._draws, axis=0, ddof=1), atol=1e-12)
    assert not list(tmp_path.iterdir())
    json.loads(r.summary_json())


def test_disk_sampler_copies_whole_histories_and_renames_duplicates(tmp_path):
    data = pl.DataFrame(
        dict(id=["a", "a", "b", "c", "c", "c"], time=[1, 3, 2, 1, 2, 3], value=np.arange(6))
    )
    raw = tmp_path / "input.parquet"
    data.write_parquet(raw)
    root = tmp_path / "stage"
    root.mkdir()
    sampler = DiskGroupResampler(
        raw,
        root,
        MemoryBudget(None),
        1,
        key="id",
        sort_columns=["id", "time"],
        select=lambda f: f,
        index_column="source_index",
        draw_column="boot",
    )
    assert sampler.n_units == 3 and sampler.n_rows == 6
    expected = np.random.default_rng(4).integers(0, 3, size=3, dtype=np.int64)
    path, dates, n, _ = sampler.materialize(
        np.random.default_rng(4), tmp_path / "draw", time_column="time", chunk_rows=2
    )
    result = pl.read_parquet(path)
    assert result["boot"].n_unique() == 3
    assert result["id"].n_unique() < 3  # this seed repeats a source unit
    for j, i in enumerate(expected):
        assert (
            result.filter(pl.col("boot") == j)
            .drop("boot")
            .sort("time")
            .equals(data.filter(pl.col("id") == ["a", "b", "c"][i]).sort("time"))
        )
    assert n == len(result) and dates == sorted(result["time"].unique().to_list())


def test_pointwise_interval_definitions_and_no_implicit_inference():
    draws = np.array([[0.0, 1.0], [1.0, 3.0], [4.0, 5.0], [9.0, 9.0], [16.0, 11.0]])
    r = RefitBootstrapResult(
        fml="y~1",
        estimator="cs",
        coefnames=["a", "b"],
        beta=np.array([3.0, 4.0]),
        diagnostics={},
        _draws=draws,
        _failures=[],
        _warnings=[],
    )
    q = np.quantile(draws, [0.1, 0.9], axis=0, method="linear")
    for method in ["percentile", "basic", "normal"]:
        tab = r.confint(0.2, method=method)
        expected = (
            q
            if method == "percentile"
            else np.array([2 * r.beta - q[1], 2 * r.beta - q[0]])
            if method == "basic"
            else np.array(
                [
                    r.beta - norm.ppf(0.9) * draws.std(axis=0, ddof=1),
                    r.beta + norm.ppf(0.9) * draws.std(axis=0, ddof=1),
                ]
            )
        )
        assert_allclose(tab["Lower"], expected[0])
        assert_allclose(tab["Upper"], expected[1])
    for method in ["bca", "studentized"]:
        with pytest.raises(ValueError, match="method"):
            r.confint(method=method)


def test_failures_are_not_redrawn_or_used_without_acknowledgment(tmp_path):
    ids = np.repeat(np.arange(6), 4)
    time = np.tile(np.arange(1, 5), 6)
    g = np.repeat([0, 3, 3, 3, 4, 4], 4)
    d = pl.DataFrame(
        dict(
            id=ids,
            time=time,
            first_treat=g,
            y=0.2 * ids + 0.3 * time + (g > 0) * (time >= g) + 0.03 * ids * time,
        )
    )
    with pytest.warns(BootstrapFailureWarning):
        r = refit_bootstrap(
            "y~1",
            d,
            reps=24,
            seed=17,
            aggregate="overall",
            estimator=did_callaway_santanna,
            effective_sample_warning=None,
            workdir=tmp_path,
            **KW,
        )
    assert 2 <= r.successful_reps < 24
    assert r.draws().height == 24 and r.failures().height == 24 - r.successful_reps
    assert any("never-treated" in e["reason"] for e in r.failures().to_dicts())
    assert len(r.diagnostics["draw_plan_sha256"]) == 24
    with pytest.raises(ValueError, match="allow_incomplete"):
        r.standard_errors()
    with pytest.raises(ValueError, match="allow_incomplete"):
        r.confint()
    assert np.isfinite(r.standard_errors(allow_incomplete=True)).all()
    assert r.tidy(allow_incomplete=True)["Successful Reps"][0] == r.successful_reps
    assert not list(tmp_path.iterdir())
    json.loads(r.summary_json())


def test_complete_panel_and_covariates_of_original_ids(tmp_path):
    data = small_data(False, True)
    # Original ID remains a covariate. Only the explicit unit FE is stripped
    # before assigning the separate bootstrap panel ID.
    options = dict(
        estimator=did_bjs,
        incomplete_units="drop",
        weights="w",
        allow_time_varying_weights=True,
    )
    a = refit_bootstrap(
        "y~x+I(id*time)|id+time",
        data,
        reps=3,
        seed=2,
        aggregate="overall",
        workdir=tmp_path,
        **KW,
        **options,
    )
    complete = data.filter(pl.len().over("id") == 5)
    b = refit_bootstrap(
        "y~x+I(id*time)",
        complete,
        reps=3,
        seed=2,
        aggregate="overall",
        workdir=tmp_path,
        **KW,
        **{**options, "incomplete_units": "error"},
    )
    assert_allclose(a._draws, b._draws, atol=2e-9, rtol=0)
    assert a.diagnostics["unit_count"] == complete["id"].n_unique()
    assert a.diagnostics["source_rows"] > a.diagnostics["analysis_rows"]
    assert not list(tmp_path.iterdir())


def test_no_never_reference_window_frozen_even_if_cohort_lost():
    data = small_data(True, False)
    r = refit_bootstrap(
        "y~1",
        data,
        reps=3,
        seed=8,
        aggregate="overall",
        estimator=did_bjs,
        **KW,
    )
    assert r.diagnostics["fixed_reference_cohort"] == 5
    assert r.diagnostics["reference_cutoff"] == 5
    assert r.diagnostics["replicate_reference"] == "never_treated"
    assert r.diagnostics["analysis_time_max"] == 4


def test_reproducibility_entropy_target_selection_and_guards(tmp_path):
    data = small_data(False, False)
    opts = dict(
        estimator=did_callaway_santanna,
        aggregate="event_time",
        effective_sample_warning=None,
        reps=3,
        workdir=tmp_path,
        **KW,
    )
    a = refit_bootstrap("y~1", data, **opts)
    b = refit_bootstrap(
        "y~1", data, seed=a.diagnostics["seed_entropy"], coefficients=a.coefnames[:2], **opts
    )
    assert_allclose(a._draws[:, :2], b._draws, atol=1e-12, rtol=0)
    assert a.diagnostics["unit_order_sha256"] == b.diagnostics["unit_order_sha256"]
    for kwargs in [
        dict(reps=1),
        dict(seed=-1),
        dict(multiplier_bootstrap_reps=4),
        dict(profile=True),
    ]:
        with pytest.raises(ValueError):
            refit_bootstrap("y~1", data, **{**opts, **kwargs})
    with pytest.raises(MemoryError):
        refit_bootstrap("y~1", data, max_memory_gb=0.01, **opts)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("error", [MemoryError, RuntimeError])
def test_resource_and_programming_errors_propagate_and_clean_scratch(error, monkeypatch, tmp_path):
    import diff_stream.bootstrap as module

    original = module._fit
    calls = 0

    def fail_after_original(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise error("injected failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_fit", fail_after_original)
    with pytest.raises(error, match="injected failure"):
        refit_bootstrap(
            "y~1",
            small_data(),
            reps=3,
            seed=2,
            estimator=did_callaway_santanna,
            aggregate="overall",
            effective_sample_warning=None,
            workdir=tmp_path,
            **KW,
        )
    assert calls == 2
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("case", ["balanced", "unbalanced_last"])
@pytest.mark.parametrize("method", ["reg", "ipw", "dr"])
def test_frozen_native_did_refits_and_base_r_intervals(case, method):
    from benchmarks.refit_bootstrap_validation import audit, provenance

    provenance()
    audit(case, method)


@pytest.mark.resource
def test_refit_bootstrap_constrained(resource_runner, resource_enforcement, tmp_path):
    code = f"""from diff_stream import did_bjs, refit_bootstrap
from diff_stream.simulate import simulate_panel
r=refit_bootstrap('y~1',simulate_panel(80,5,seed=17),panel=('id','time'),cohort='first_treat',estimator=did_bjs,reps=3,seed=42,aggregate='overall',n_threads=2,preprocess_mb=4,workdir={str(tmp_path / "child")!r})
assert r.successful_reps==3
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
    if result.kernel:
        assert result.kernel["events"]["oom_kill"] == 0
    assert not list((tmp_path / "child").iterdir())
