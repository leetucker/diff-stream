"""The refit bootstrap of did_dcdh: draws of groups (or of clusters, with cluster=)."""

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from benchmarks._files import digest, read_csv
from diff_stream import ContinuousBaselineWarning, did_dcdh, refit_bootstrap

DATA = Path(__file__).parent / "data"
KW = dict(panel=("id", "time"), treatment="d", n_threads=2, preprocess_mb=4, batch_units=37)
FIT = dict(effects=3, placebos=2)


def switching(panel="switching"):
    frame = read_csv(DATA / "dcdh_panel.csv").filter(pl.col("panel") == panel).drop("panel")
    return frame.with_columns(cl=pl.col("id") // 8)


def _fit(rows, **options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ContinuousBaselineWarning)
        return did_dcdh("y ~ x + z", rows, **(KW | dict(panel=("boot", "time"))), **FIT, **options)


@pytest.mark.parametrize(
    "panel,options",
    [
        ("switching", {}),
        ("unbalanced", dict(incomplete_units="keep", weights="w", allow_time_varying_weights=True)),
    ],
)
def test_every_draw_matches_explicit_group_duplication(panel, options, tmp_path):
    data = switching(panel)
    r = refit_bootstrap(
        "y ~ x + z",
        data,
        estimator=did_dcdh,
        reps=4,
        seed=29,
        workdir=tmp_path,
        **KW,
        **FIT,
        **options,
    )
    assert not r.failures().height
    assert r.diagnostics["resampled"] == "units"
    units = sorted(data["id"].unique().to_list())
    rng = np.random.default_rng(29)
    for j in range(4):
        indices = rng.integers(0, len(units), size=len(units), dtype=np.int64)
        assert (
            r.diagnostics["draw_plan_sha256"][j]
            == hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest()
        )
        rows = pl.concat(
            [
                data.filter(pl.col("id") == units[int(u)]).with_columns(pl.lit(i).alias("boot"))
                for i, u in enumerate(indices)
            ]
        )
        expected = _fit(rows, **options)
        assert_allclose(r._draws[j], expected.beta, atol=2e-9, rtol=0)
    assert r.coefnames == did_dcdh("y ~ x + z", data, **KW, **FIT, **options).coefnames
    assert not list(tmp_path.iterdir())
    json.loads(r.summary_json())


def test_clustered_draws_resample_whole_clusters(tmp_path):
    """With cluster=, did_multiplegt_dyn's bootstrap draws clusters; so does this one,
    and each copy's groups get IDs of their own."""
    data = switching()
    r = refit_bootstrap(
        "y ~ x + z",
        data,
        estimator=did_dcdh,
        cluster="cl",
        reps=3,
        seed=5,
        workdir=tmp_path,
        **KW,
        **FIT,
    )
    assert not r.failures().height
    assert r.diagnostics["resampled"] == "clusters (cl)" and r.diagnostics["clusters"] == 50
    assert r.diagnostics["sampling"].startswith("ordinary uniform clusters of cl")
    clusters = sorted(data["cl"].unique().to_list())
    size = data.group_by("cl").agg(pl.col("id").n_unique())["id"].max()
    rng = np.random.default_rng(5)
    for j in range(3):
        indices = rng.integers(0, len(clusters), size=len(clusters), dtype=np.int64)
        copies = []
        for copy, c in enumerate(indices):
            members = data.filter(pl.col("cl") == clusters[int(c)])
            rank = members["id"].rank("dense").cast(pl.Int64) - 1
            copies.append(members.with_columns(boot=copy * size + rank))
        expected = _fit(pl.concat(copies))
        assert_allclose(r._draws[j], expected.beta, atol=2e-9, rtol=0)


def test_parallel_draws_match_serial(tmp_path):
    data = switching()
    options = dict(estimator=did_dcdh, cluster="cl", reps=4, seed=11, **KW, **FIT)
    serial = refit_bootstrap("y ~ x + z", data, workdir=tmp_path, **options)
    parallel = refit_bootstrap(
        "y ~ x + z", data, workdir=tmp_path, n_workers=2, max_memory_gb=6, **options
    )
    assert_allclose(parallel._draws, serial._draws, rtol=1e-12, atol=1e-14)
    assert parallel.diagnostics["draw_plan_sha256"] == serial.diagnostics["draw_plan_sha256"]
    assert not list(tmp_path.iterdir())


def test_bootstrap_standard_errors_are_close_to_the_analytic_ones(tmp_path):
    """Without covariates and clusters, the authors' analytic standard errors are
    conservative but close; 199 draws estimate a standard error to about 5%."""
    data = switching()
    analytic = did_dcdh("y ~ 1", data, **KW, effects=2, placebos=1)
    r = refit_bootstrap(
        "y ~ 1",
        data,
        estimator=did_dcdh,
        reps=199,
        seed=3,
        workdir=tmp_path,
        effects=2,
        placebos=1,
        **KW,
    )
    ratio = r.se / analytic.se
    assert np.all((ratio > 0.75) & (ratio < 1.25)), ratio


def test_continuous_baseline_and_errors(tmp_path):
    data = read_csv(DATA / "dcdh_continuous_panel.csv").filter(pl.col("panel") == "continuous")
    data = data.drop("panel")
    with pytest.warns(ContinuousBaselineWarning):
        r = refit_bootstrap(
            "y ~ 1",
            data,
            estimator=did_dcdh,
            continuous_baseline=2,
            reps=3,
            seed=1,
            workdir=tmp_path,
            **KW,
            **FIT,
        )
    assert np.isfinite(r._draws).all()
    # each refit warns too; the warnings are kept with the draws
    assert set(r.refit_warnings()["category"]) == {"ContinuousBaselineWarning"}
    with pytest.raises(ValueError, match="reported directly"):
        refit_bootstrap(
            "y ~ 1",
            data,
            estimator=did_dcdh,
            aggregate="overall",
            reps=2,
            continuous_baseline=2,
            **KW,
        )
    with pytest.raises(TypeError, match="has no option"):
        refit_bootstrap("y ~ 1", data, estimator=did_dcdh, reps=2, cohort_size=3, **KW)


BOOT = DATA / "dcdh_bootstrap_reference"


@pytest.mark.parametrize("cluster", [None, "cl"])
def test_bootstrap_standard_errors_agree_with_did_multiplegt_dyn(cluster, tmp_path):
    """did_multiplegt_dyn's bootstrap (500 draws, its own random numbers) against 500
    of these draws: two independent bootstrap standard errors differ by about 4.5%
    (one standard deviation), so they must agree within 15%."""
    for name, h in json.loads((BOOT / "PROVENANCE.json").read_text())["files"].items():
        assert digest(Path(__file__).parents[1] / name) == h, name
    ref = read_csv(BOOT / "bootstrap.csv").filter(pl.col("cluster") == (cluster or "none"))
    r = refit_bootstrap(
        "y ~ 1",
        switching(),
        estimator=did_dcdh,
        cluster=cluster,
        reps=500,
        seed=1,
        workdir=tmp_path,
        effects=2,
        placebos=1,
        **KW,
    )
    coefficients = ref["coefficient"].to_list()
    se = r.se[[r.coefnames.index(c) for c in coefficients]]
    estimate = r.beta[[r.coefnames.index(c) for c in coefficients]]
    assert_allclose(estimate, ref["estimate"].to_numpy(), rtol=1e-10)
    assert_allclose(se, ref["se"].to_numpy(), rtol=0.15)
