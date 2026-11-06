"""Public Castle Doctrine SA reanalysis: sample, timing and full covariance."""

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from benchmarks._files import read_csv
from benchmarks.castle_replication import (
    DATA,
    REF,
    compare_replication,
    prepare_data,
    verify_provenance,
)
from diff_stream import did_sun_abraham


def test_frozen_sample_and_both_explicit_treatment_conventions():
    verify_provenance()
    raw = read_csv(DATA, null_values="NA")
    assert raw.columns == ["year", "state", "popwt", "l_homicide", "post", "effyear", "time_til"]
    assert raw.shape == (550, 7) and raw["state"].n_unique() == 50
    assert sorted(raw["year"].unique().to_list()) == list(range(2000, 2011))
    assert raw.select(pl.struct("state", "year").n_unique()).item() == 550
    assert raw.group_by("state").len()["len"].to_list() == [11] * 50
    assert raw["effyear"].null_count() == 319
    assert raw.group_by("state").agg(pl.col("popwt").n_unique())["popwt"].max() == 1
    for shift in (0, 1):
        d = prepare_data(raw, shift)
        assert d.drop("first_treat").equals(raw)
        assert d.filter(pl.col("first_treat") == 0)["state"].n_unique() == 29
        assert d.filter(pl.col("first_treat") > 0)["state"].n_unique() == 21
        post = (d["first_treat"] > 0) & (d["year"] >= d["first_treat"])
        assert int((d["post"] != post).sum()) == (21 if shift == 0 else 0)
    assert sorted(prepare_data(raw)["first_treat"].unique().to_list()) == [
        0,
        2005,
        2006,
        2007,
        2008,
        2009,
    ]


@pytest.mark.parametrize("shift", [0, 1])
@pytest.mark.parametrize("weighted", [False, True])
def test_actual_fixest_aggregations_and_independent_dummy_oracles(shift, weighted):
    fit, differences = compare_replication(shift, weighted)
    try:
        assert max(differences.values()) < 1e-8
        assert len(fit.cells) == 50
        assert fit.diagnostics["dropped_units"] == fit.diagnostics["dropped_rows"] == 0
        assert all(t != g - 1 for g, t in fit.cells)
    finally:
        fit.cleanup()


@pytest.mark.parametrize("weighted", [False, True])
def test_layout_batch_and_crv_switching(tmp_path, weighted):
    baseline, _ = compare_replication(weighted=weighted, dense=False)
    d = prepare_data().sample(fraction=1, shuffle=True, seed=811)
    d.write_parquet(tmp_path / "castle.parquet", row_group_size=13)
    try:
        for source in (d.lazy(), str(tmp_path / "castle.parquet")):
            fit = did_sun_abraham(
                "l_homicide ~ 1",
                source,
                panel=("state", "year"),
                cohort="first_treat",
                weights="popwt" if weighted else None,
                n_threads=2,
                vcov="CRV1",
                batch_units=7,
                preprocess_mb=1,
                workdir=tmp_path / "work",
            )
            try:
                assert fit.cells == baseline.cells
                assert_allclose(fit.beta, baseline.beta, atol=1e-10)
                assert_allclose(fit.vcov, baseline.vcov * 50 / 49, atol=1e-10)
                assert_allclose(
                    fit.aggregate().vcov, baseline.aggregate().vcov * 50 / 49, atol=1e-10
                )
                assert_allclose(
                    fit.with_vcov("CRV0").aggregate("overall").se,
                    baseline.aggregate("overall").se,
                    atol=1e-10,
                )
            finally:
                fit.cleanup()
            assert not list((tmp_path / "work").iterdir())
    finally:
        baseline.cleanup()


def test_reporting_window_preserves_saturation_and_exposes_thin_support():
    fit, _ = compare_replication(dense=False)
    try:
        full = fit.beta.copy()
        narrowed = fit.aggregate(horizons=[0, 1, 2])
        assert len(fit.cells) == 50
        assert_allclose(fit.beta, full, atol=0, rtol=0)
        reference = read_csv(REF / "shift0_unweighted_dynamic.csv")
        ix = [reference["level"].to_list().index(h) for h in [0, 1, 2]]
        assert_allclose(narrowed.beta, reference["att"].to_numpy()[ix], atol=1e-10)
        contributions = fit.aggregate().contributions()
        for event in (-9, 5):
            rows = contributions.filter(pl.col("event_time") == event)
            assert rows["weight_units"].sum() == 1
            assert_allclose(rows["weight"].sum(), 1, atol=1e-12)
        assert (
            fit.aggregate(balanced_horizon=2, horizons=[0, 1, 2])
            .composition()["cohorts_changed"]
            .not_()
            .all()
        )
        assert np.isfinite(fit.aggregate().vcov).all()
    finally:
        fit.cleanup()
