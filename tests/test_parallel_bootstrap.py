"""Process isolation, deterministic draws and whole-job parallel resource policy."""

import os
import sys
import time
from concurrent.futures import as_completed

import numpy as np
import pytest
from numpy.testing import assert_allclose
from stream_limits import current_rss_gb
from test_refit_bootstrap import KW, small_data

from benchmarks._estimators import ESTIMATORS
from diff_stream import BootstrapFailureWarning, did_callaway_santanna, refit_bootstrap
from diff_stream._core.process import ProcessBatch


@pytest.mark.parametrize("estimator", ["cs", "sunab", "bjs", "gardner", "etwfe"])
@pytest.mark.parametrize("last", [False, True])
def test_parallel_draws_match_serial_and_use_divided_threads(estimator, last, tmp_path):
    data = small_data(last, True)
    options = dict(
        weights="w",
        allow_time_varying_weights=True,
        incomplete_units="keep",
        anticipation=1,
        max_memory_gb=6,
        estimator=ESTIMATORS[estimator],
    )
    if estimator == "cs":
        options["effective_sample_warning"] = None
    target = None if estimator == "gardner" else "event_time"
    args = dict(reps=3, seed=29, aggregate=target, workdir=tmp_path, **KW, **options)
    serial = refit_bootstrap("y~x", data, **args)
    parallel = refit_bootstrap("y~x", data, n_workers=2, **args)
    assert_allclose(parallel._draws, serial._draws, atol=2e-9, rtol=0)
    assert_allclose(parallel.se, serial.se, atol=2e-9, rtol=0)
    for method in ["percentile", "basic", "normal"]:
        assert_allclose(
            parallel.confint(method=method).select("Lower", "Upper").to_numpy(),
            serial.confint(method=method).select("Lower", "Upper").to_numpy(),
            atol=3e-9,
            rtol=0,
        )
    assert parallel.diagnostics["draw_plan_sha256"] == serial.diagnostics["draw_plan_sha256"]
    assert parallel.diagnostics["unit_order_sha256"] == serial.diagnostics["unit_order_sha256"]
    assert parallel.failures().equals(serial.failures())
    assert parallel.refit_warnings().equals(serial.refit_warnings())
    assert parallel.diagnostics["worker_threads"] == 1
    resources = parallel.diagnostics["worker_resources"]
    assert [v["draw"] for v in resources] == [0, 1, 2]
    assert all(v["polars_threads"] == 1 for v in resources)
    assert all(v["pid"] != os.getpid() for v in resources)
    assert [v["refits"] for v in parallel.diagnostics["parallel_batches"]] == [2, 1]
    assert not list(tmp_path.iterdir())


def test_parallel_failed_attempts_match_serial_without_redrawing(tmp_path):
    import polars as pl

    ids, times = np.repeat(np.arange(6), 4), np.tile(np.arange(1, 5), 6)
    g = np.repeat([0, 3, 3, 3, 4, 4], 4)
    data = pl.DataFrame(
        dict(
            id=ids,
            time=times,
            first_treat=g,
            y=0.2 * ids + 0.3 * times + (g > 0) * (times >= g) + 0.03 * ids * times,
        )
    )
    options = dict(
        reps=8,
        seed=2,
        estimator=did_callaway_santanna,
        aggregate="overall",
        effective_sample_warning=None,
        max_memory_gb=6,
        workdir=tmp_path,
        **KW,
    )
    with pytest.warns(BootstrapFailureWarning):
        serial = refit_bootstrap("y~1", data, **options)
    with pytest.warns(BootstrapFailureWarning):
        parallel = refit_bootstrap("y~1", data, n_workers=2, **options)
    assert 0 < parallel.failures().height < 8
    assert_allclose(parallel._draws, serial._draws, atol=2e-9, rtol=0, equal_nan=True)
    assert parallel.failures().equals(serial.failures())
    assert parallel.diagnostics["draw_plan_sha256"] == serial.diagnostics["draw_plan_sha256"]
    with pytest.raises(ValueError, match="allow_incomplete"):
        parallel.confint()
    assert not list(tmp_path.iterdir())


def test_parallel_validates_resources_before_starting_workers(tmp_path):
    data = small_data()
    options = dict(
        reps=2,
        estimator=did_callaway_santanna,
        aggregate="overall",
        effective_sample_warning=None,
        workdir=tmp_path,
        **KW,
    )
    for workers in [0, -1, True, 1.5]:
        with pytest.raises(ValueError, match="workers"):
            refit_bootstrap("y~1", data, n_workers=workers, **options)
    with pytest.raises(ValueError, match="thread budget"):
        refit_bootstrap("y~1", data, n_workers=2, **{**options, "n_threads": 1})
    with pytest.raises(MemoryError, match="whole-job memory"):
        # room for the parent's staging, not for two workers, whatever this process holds
        limit = (current_rss_gb() or 0) + 0.9
        refit_bootstrap("y~1", data, n_workers=2, max_memory_gb=limit, **options)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("failure", ["programming", "memory"])
def test_remote_errors_propagate_and_clean_scratch(failure, monkeypatch, tmp_path):
    import diff_stream.bootstrap as module

    original = module._parallel_refits

    def invalid_task(tasks, *args):
        if failure == "programming":
            # A genuine subprocess KeyError must abort, not become a failed draw.
            del tasks[0]["fml"]
        else:
            args = (*args[:-1], 0.1)  # Force the child estimator's MemoryBudget guard.
        return original(tasks, *args)

    monkeypatch.setattr(module, "_parallel_refits", invalid_task)
    with pytest.raises(KeyError if failure == "programming" else MemoryError):
        refit_bootstrap(
            "y~1",
            small_data(),
            n_workers=2,
            reps=2,
            max_memory_gb=6,
            estimator=did_callaway_santanna,
            aggregate="overall",
            effective_sample_warning=None,
            workdir=tmp_path,
            **KW,
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_process_batch_stops_running_sibling_on_failure_or_interrupt(error, tmp_path):
    ready = tmp_path / "ready"
    processes = []
    start = time.monotonic()
    with pytest.raises(error):
        with ProcessBatch(2) as batch:
            sleeper = batch.submit(
                [
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; import time; Path({str(ready)!r}).touch(); time.sleep(60)",
                ],
                tmp_path / "sleep.log",
            )
            deadline = time.monotonic() + 10
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert ready.exists()
            with batch._lock:
                processes = list(batch._children)
            if error is KeyboardInterrupt:
                raise KeyboardInterrupt()
            bad = batch.submit([sys.executable, "-c", "raise SystemExit(42)"], tmp_path / "bad.log")
            for future in as_completed([sleeper, bad]):
                if future.result():
                    raise RuntimeError("worker failed")
    assert time.monotonic() - start < 15
    assert processes and all(p.poll() is not None for p in processes)


@pytest.mark.resource
def test_parallel_bootstrap_constrained(resource_runner, resource_enforcement, tmp_path):
    code = f"""from diff_stream import did_bjs, refit_bootstrap
from diff_stream.simulate import simulate_panel
r=refit_bootstrap('y~1',simulate_panel(80,5,seed=17),panel=('id','time'),cohort='first_treat',estimator=did_bjs,n_workers=2,reps=3,seed=42,aggregate='overall',n_threads=2,max_memory_gb=2.5,preprocess_mb=4,workdir={str(tmp_path / "child")!r})
assert r.successful_reps==3
assert all(w['polars_threads']==1 for w in r.diagnostics['worker_resources'])
print(r.summary_json())
"""
    env = dict(
        os.environ,
        STREAM_LIMITS_MAX_THREADS="2",
        POLARS_MAX_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        MPLCONFIGDIR=str(tmp_path / "mpl"),
    )
    r = resource_runner([sys.executable, "-c", code], memory_gb=2.5, timeout=90, env=env)
    assert r.status == "ok", r.stderr
    assert r.mode == resource_enforcement["mode"]
    if r.kernel:
        assert r.kernel["events"]["oom_kill"] == 0
    assert not list((tmp_path / "child").iterdir())
