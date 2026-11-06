"""Inference edge cases and invariants of bounded whole-unit preprocessing."""

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from diff_stream import did_bjs, did_callaway_santanna
from diff_stream.simulate import simulate_panel

KW = dict(panel=("id", "time"), cohort="first_treat", n_threads=2)


def test_leaveout_changes_covariance_and_preserves_crv_scaling(data):
    a = did_bjs("y ~ 1", data, **KW)
    b = did_bjs("y ~ 1", data, leave_out=True, **KW)
    assert_allclose(a.beta, b.beta)
    # With larger cohorts the adjustment is small but nonzero.
    assert np.max(np.abs(a.vcov - b.vcov)) > 1e-5
    assert_allclose(
        b.with_vcov("CRV1").aggregate().vcov, b.aggregate().vcov * b.n_units / (b.n_units - 1)
    )


def test_singleton_and_broader_residual_groups():
    d = simulate_panel(40, 4)
    # Exactly one unit in the first treatment cohort; others are never or g=3.
    d = d.with_columns(
        pl.when(pl.col("id") == 0)
        .then(2)
        .when(pl.col("id") < 20)
        .then(0)
        .otherwise(3)
        .alias("first_treat")
    )
    with pytest.raises(ValueError, match="leave_out requires"):
        did_bjs("y ~ 1", d, leave_out=True, **KW)
    r = did_bjs("y ~ 1", d, leave_out=True, residual_averaging="all", **KW)
    assert np.isnan(r.se).any()
    a = r.aggregate("overall")
    assert np.isfinite(a.se).all()
    assert a.se[0] > 0


def test_smart_aggregation_is_contrast_specific(data):
    r = did_bjs("y ~ 1", data, residual_averaging="all", leave_out=True, **KW)
    A = r.counts / r.counts.sum()
    a = r.aggregate("overall")
    assert_allclose(a.beta, A @ r.beta)
    assert abs(a.vcov[0, 0] - A @ r.vcov @ A) > 1e-6
    # Subsetting horizons also recomputes the correct smart weights.
    s = r.aggregate(horizons=[0, 1])
    assert_allclose(s.beta, r.aggregate().beta[:2])
    assert_allclose(s.vcov, r.aggregate().vcov[:2, :2])


@pytest.mark.parametrize("fit", [did_callaway_santanna, did_bjs])
def test_partitioned_results_match_single_leaf(fit):
    d = simulate_panel(5000, 5, seed=887)
    a = fit("y ~ 1", d, preprocess_mb=64, **KW)
    b = fit(
        "y ~ 1", d.sample(fraction=1, shuffle=True, seed=14), preprocess_mb=1, batch_units=31, **KW
    )
    assert b.diagnostics["partitions"] > 1
    assert b.diagnostics["max_decoded_leaf_bytes"] <= 250000
    assert_allclose(a.beta, b.beta, atol=1e-11)
    assert_allclose(a.vcov, b.vcov, atol=1e-11)


def test_recursive_partition_budget():
    p = did_callaway_santanna(
        "y ~ 1", simulate_panel(50000, 8), preprocess_mb=1, profile=True, **KW
    )
    assert p.notes["max_partition_depth"] >= 2
    assert p.notes["max_decoded_leaf_bytes"] <= 250000
    assert p.notes["n_units"] == 50000


def test_budget_fails_before_source_is_executed(data, monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise AssertionError("source executed before budget check")

    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", fail)
    with pytest.raises(MemoryError, match="before staging"):
        did_callaway_santanna("y ~ 1", data, max_memory_gb=0.001, workdir=tmp_path, **KW)
    assert list(tmp_path.iterdir()) == []


def test_extreme_unit_skew_fails_without_oversized_sort(tmp_path):
    d = pl.DataFrame(
        {
            "id": np.zeros(20000, dtype=int),
            "time": np.tile([1, 2], 10000),
            "first_treat": np.zeros(20000, dtype=int),
            "y": np.ones(20000),
        }
    )
    with pytest.raises(ValueError, match="balanced"):
        did_callaway_santanna("y ~ 1", d, preprocess_mb=1, workdir=tmp_path, **KW)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"preprocess_mb": 0},
        {"preprocess_mb": float("inf")},
        {"leave_out": "yes"},
        {"residual_averaging": "bad"},
    ],
)
def test_new_option_validation(data, kwargs):
    with pytest.raises(ValueError):
        did_bjs("y ~ 1", data, **KW, **kwargs)
