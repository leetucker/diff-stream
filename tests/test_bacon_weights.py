"""Weighted Bacon: exact package replication and independent original-row WLS."""

import json

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose
from test_bacon import CASES, KW, ROOT, fixture

from benchmarks._estimators import public
from benchmarks._files import read_csv
from diff_stream import bacon_decomp
from diff_stream.simulate import simulate_panel
from diff_stream.store import PanelStore

REF = ROOT / "bacon_weights_reference"


def weighted_panel(case, kind="rational"):
    d = fixture(case)
    w = (1 + 5 * (pl.col("id") % 9)) / 5 if kind == "rational" else (2 + pl.col("id") % 9).sqrt()
    return d.with_columns(w.alias("w"))


def retained(d, case):
    return d.filter(pl.col("time").count().over("id") == 9) if case == "complete" else d


def dense_wls(d):
    ids, t, g, y, w = (d[c].to_numpy() for c in ("id", "time", "first_treat", "y", "w"))
    X = np.column_stack(
        [
            (g > 0) & (t >= g),
            *[ids == u for u in np.unique(ids)],
            *[t == v for v in np.unique(t)[1:]],
        ]
    ).astype(float)
    root = np.sqrt(w / w.max())
    return np.linalg.lstsq(X * root[:, None], y * root, rcond=None)[0][0]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("kind", ["rational", "irrational"])
def test_r_weighted_twfe_and_comparisons(case, kind):
    d = weighted_panel(case, kind)
    r = bacon_decomp(
        "y ~ 1",
        d,
        weights="w",
        incomplete_units=public(
            "incomplete_units", "complete" if case == "complete" else "balanced"
        ),
        **KW,
    )
    rows = r.comparisons().sort(["treated_cohort", "control_cohort"])
    ref = read_csv(REF / f"{case}_{kind}.csv").sort(["treated", "untreated"])
    assert rows["treated_cohort"].to_list() == ref["treated"].to_list()
    assert rows["control_cohort"].to_list() == ref["untreated"].to_list()
    assert_allclose(rows["estimate"], ref["estimate"], atol=3e-12)
    if kind == "rational":
        assert_allclose(rows["weight"], ref["weight"], atol=3e-14)
        assert_allclose(rows["contribution"], ref["weight"] * ref["estimate"], atol=3e-12)
    check = (
        read_csv(REF / "twfe.csv")
        .filter((pl.col("case") == case) & (pl.col("weights") == kind))
        .row(0, named=True)
    )
    used = retained(d, case)
    assert_allclose(r.twfe_estimate, check["twfe"], atol=3e-12)
    assert_allclose(r.twfe_estimate, dense_wls(used), atol=3e-12)
    assert_allclose(r.decomposed_estimate, r.twfe_estimate, atol=2e-12)
    assert_allclose(rows["weight"].sum(), 1, atol=1e-14)
    assert (rows["weight"] > 0).all()
    assert r.n_units == check["n_units"] and r.n_obs == check["n_obs"]
    assert_allclose(r.diagnostics["total_unit_weight"], check["total_unit_weight"], rtol=1e-13)
    normalized = used.with_columns(
        pl.when(pl.col("first_treat") > used["time"].max())
        .then(0)
        .otherwise(pl.col("first_treat"))
        .alias("first_treat")
    )
    for row in rows.iter_rows(named=True):
        for role in ("treated", "control"):
            units = normalized.filter(pl.col("first_treat") == row[f"{role}_cohort"]).unique("id")
            assert row[f"n_{role}_units"] == len(units)
            assert_allclose(row[f"{role}_weight_total"], units["w"].sum(), atol=2e-12)
        pair = normalized.filter(
            pl.col("first_treat").is_in([row["treated_cohort"], row["control_cohort"]])
            & pl.col("time").is_between(row["pre_start"], row["post_end"])
        )
        pair = pair.with_columns(
            pl.when(pl.col("first_treat") == row["treated_cohort"])
            .then(row["post_start"])
            .otherwise(0)
            .alias("first_treat")
        )
        assert_allclose(row["estimate"], dense_wls(pair), atol=4e-12)
    assert_allclose(r.summary()["contribution"].sum(), r.twfe_estimate, atol=3e-12)
    assert_allclose(r.summary()["weight"].sum(), 1, atol=1e-14)


@pytest.mark.parametrize("case", CASES)
def test_equal_weights_and_common_rescaling(case, tmp_path):
    d = weighted_panel(case)
    mode = "complete" if case == "complete" else "balanced"
    a = bacon_decomp(
        "y ~ 1",
        d,
        weights="w",
        incomplete_units=public("incomplete_units", mode),
        workdir=tmp_path,
        **KW,
    )
    for scale in (1e-200, 1e200):
        scaled = d.with_columns(pl.col("w") * scale).sample(fraction=1, shuffle=True, seed=19)
        b = bacon_decomp(
            "y ~ 1",
            scaled,
            weights="w",
            incomplete_units=public("incomplete_units", mode),
            preprocess_mb=1,
            workdir=tmp_path,
            **KW,
        )
        assert_allclose(a.twfe_estimate, b.twfe_estimate, atol=2e-12)
        assert_allclose(a.comparisons()["estimate"], b.comparisons()["estimate"], atol=2e-12)
        assert_allclose(a.comparisons()["weight"], b.comparisons()["weight"], atol=2e-14)
        assert_allclose(
            a.comparisons()["treated_weight_total"] * scale,
            b.comparisons()["treated_weight_total"],
            rtol=1e-13,
        )
        assert_allclose(
            a.diagnostics["total_unit_weight"] * scale,
            b.diagnostics["total_unit_weight"],
            rtol=1e-13,
        )
    plain = bacon_decomp("y ~ 1", d, incomplete_units=public("incomplete_units", mode), **KW)
    equal = bacon_decomp(
        "y ~ 1",
        d.with_columns(pl.lit(7.3).alias("w")),
        weights="w",
        incomplete_units=public("incomplete_units", mode),
        **KW,
    )
    assert_allclose(plain.twfe_estimate, equal.twfe_estimate, atol=1e-12)
    assert_allclose(plain.comparisons()["estimate"], equal.comparisons()["estimate"], atol=1e-12)
    assert_allclose(plain.comparisons()["weight"], equal.comparisons()["weight"], atol=2e-14)
    assert_allclose(
        equal.comparisons()["treated_weight_total"],
        7.3 * equal.comparisons()["n_treated_units"],
        rtol=1e-13,
    )
    assert not list(tmp_path.iterdir())


def test_source_types_partition_sizes_and_string_ids(tmp_path):
    d = simulate_panel(1300, 8, seed=215).with_columns((0.2 + pl.col("id") % 9).alias("w"))
    a = bacon_decomp("y ~ 1", d, weights="w", preprocess_mb=1, **KW)
    assert a.diagnostics["partitions"] > 1
    d[:5000].write_parquet(tmp_path / "a.parquet", row_group_size=7)
    d[5000:].write_parquet(tmp_path / "b.parquet", row_group_size=31)
    b = bacon_decomp(
        "y ~ 1",
        str(tmp_path / "*.parquet"),
        weights="w",
        preprocess_mb=2,
        workdir=tmp_path / "work",
        **KW,
    )
    c = bacon_decomp(
        "y ~ 1", d.with_columns(pl.col("id").cast(pl.String)).lazy(), weights="w", **KW
    )
    for r in (b, c):
        assert_allclose(a.twfe_estimate, r.twfe_estimate, atol=1e-12)
        assert_allclose(a.comparisons()["estimate"], r.comparisons()["estimate"], atol=1e-12)
        assert_allclose(a.comparisons()["weight"], r.comparisons()["weight"], atol=1e-14)
    assert not list((tmp_path / "work").iterdir())
    report = json.loads(a.summary_json(comparisons=True))
    assert report["diagnostics"]["sampling_weights"] == "w"
    assert report["n_units"] == 1300
    assert len(report["comparisons"]) == 49


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("-inf"), float("nan"), None])
def test_invalid_values_and_cleanup(bad, tmp_path):
    d = fixture("heterogeneous").with_columns(pl.lit(bad, dtype=pl.Float64).alias("w"))
    with pytest.raises(ValueError, match="positive|finite|missing"):
        bacon_decomp("y ~ 1", d, weights="w", workdir=tmp_path, **KW)
    assert not list(tmp_path.iterdir())


def test_varying_weights_and_schema_rejection(tmp_path, monkeypatch):
    d = fixture("complete").with_columns(pl.col("time").cast(pl.Float64).alias("w"))
    with pytest.raises(ValueError, match="constant within unit"):
        bacon_decomp("y ~ 1", d, weights="w", incomplete_units="drop", workdir=tmp_path, **KW)

    def no_source(*args, **kwargs):
        raise AssertionError("executed rows")

    monkeypatch.setattr(PanelStore, "__init__", no_source)
    for expression in (pl.lit("bad"), pl.lit(True)):
        with pytest.raises(ValueError, match="numeric"):
            bacon_decomp("y ~ 1", d.with_columns(expression.alias("w")), weights="w", **KW)
    for name in (True, 12, [], "", "time", "y"):
        with pytest.raises(ValueError, match="column name|distinct"):
            bacon_decomp("y ~ 1", d, weights=name, **KW)
    with pytest.raises(ValueError, match="balanced panel"):
        bacon_decomp("y ~ 1", d, weights="w", incomplete_units="keep", **KW)
    assert not list(tmp_path.iterdir())


def test_profile_and_weighted_complete_deletion(monkeypatch):
    from diff_stream import bacon, bacon_decomp

    d = weighted_panel("complete")
    clean = retained(d, "complete")
    a = bacon_decomp("y ~ 1", d, weights="w", incomplete_units="drop", **KW)
    b = bacon_decomp("y ~ 1", clean, weights="w", **KW)
    assert_allclose(a.twfe_estimate, b.twfe_estimate, atol=1e-12)
    assert_allclose(a.comparisons()["weight"], b.comparisons()["weight"], atol=1e-14)
    assert a.diagnostics["dropped_units"] == 6
    assert_allclose(a.diagnostics["total_unit_weight"], clean.unique("id")["w"].sum())

    def no_table(*args, **kwargs):
        raise AssertionError("constructed table while profiling")

    monkeypatch.setattr(bacon, "_decompose", no_table)
    p = bacon_decomp("y ~ 1", d, weights="w", incomplete_units="drop", profile=True, **KW)
    assert p.notes["sampling_weights"] == "w"
    assert_allclose(p.notes["total_unit_weight"], a.diagnostics["total_unit_weight"])
    assert p.notes["n_comparisons"] == 9
