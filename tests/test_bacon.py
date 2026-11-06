"""Goodman–Bacon comparison references, dense regressions and bounded input."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from benchmarks._files import read_csv
from diff_stream import BaconResult, bacon_decomp, cs_stream
from diff_stream.simulate import simulate_panel
from diff_stream.store import PanelStore

ROOT = Path(__file__).parent / "data"
KW = dict(panel=("id", "time"), cohort="first_treat", threads=2)
CASES = [
    "heterogeneous",
    "no_never",
    "single_timing",
    "future",
    "shifted",
    "complete",
    "constant",
    "null",
    "two_period",
]


def fixture(case):
    d = read_csv(ROOT / "bacon_panel.csv")
    if case == "no_never":
        d = d.filter(pl.col("first_treat") != 0)
    elif case == "single_timing":
        d = d.filter(pl.col("first_treat").is_in([0, 5]))
    elif case == "future":
        d = d.with_columns(
            pl.when(pl.col("first_treat") == 0)
            .then(20)
            .otherwise(pl.col("first_treat"))
            .alias("first_treat")
        )
    elif case == "shifted":
        d = d.with_columns(
            pl.col("time") + 20,
            pl.when(pl.col("first_treat") > 0)
            .then(pl.col("first_treat") + 20)
            .otherwise(0)
            .alias("first_treat"),
        )
    elif case == "complete":
        d = d.filter(~((pl.col("id") % 4 == 0) & pl.col("time").is_in([3, 7])))
    elif case in ("constant", "null"):
        d = d.with_columns(pl.col("constant" if case == "constant" else "baseline").alias("y"))
    elif case == "two_period":
        d = d.filter((pl.col("first_treat") != 2) & pl.col("time").is_in([4, 5])).with_columns(
            pl.col("time") - 3,
            pl.when(pl.col("first_treat") > 0)
            .then(pl.col("first_treat") - 3)
            .otherwise(0)
            .alias("first_treat"),
        )
    return d


def dense_twfe(d):
    """Explicit full row/unit/time regression, no production transformations."""
    ids, t, g, y = (d[c].to_numpy() for c in ("id", "time", "first_treat", "y"))
    D = (g > 0) & (t >= g)
    X = np.column_stack(
        [D, *[ids == u for u in np.unique(ids)], *[t == v for v in np.unique(t)[1:]]]
    ).astype(float)
    return np.linalg.lstsq(X, y, rcond=None)[0][0]


@pytest.mark.parametrize("case", CASES)
def test_r_bacondecomp_and_full_twfe(case):
    d = fixture(case)
    mode = "complete" if case == "complete" else "balanced"
    r = bacon_decomp("y ~ 1", d, panel_mode=mode, **KW)
    assert isinstance(r, BaconResult)
    rows = r.comparisons().sort(["treated_cohort", "control_cohort"])
    ref = read_csv(ROOT / "bacon_reference" / f"{case}.csv").sort(["treated", "untreated"])
    assert rows["treated_cohort"].to_list() == ref["treated"].to_list()
    assert rows["control_cohort"].to_list() == ref["untreated"].to_list()
    assert_allclose(rows["estimate"], ref["estimate"], atol=2e-12)
    assert_allclose(rows["weight"], ref["weight"], atol=2e-14)
    types = {
        "Treated vs Untreated": "treated_vs_never",
        "Earlier vs Later Treated": "earlier_vs_later",
        "Later vs Earlier Treated": "later_vs_earlier",
    }
    assert rows["comparison_type"].to_list() == [types[s] for s in ref["type"]]
    twfe = (
        read_csv(ROOT / "bacon_reference/twfe.csv")
        .filter(pl.col("case") == case)
        .row(0, named=True)
    )
    assert_allclose(r.twfe_estimate, twfe["twfe"], atol=2e-12)
    assert_allclose(r.decomposed_estimate, twfe["decomposed"], atol=2e-12)
    assert r.n_obs == twfe["n_obs"] and r.n_units == twfe["n_units"]
    assert_allclose(rows["weight"].sum(), 1, atol=1e-14)
    assert (rows["weight"] > 0).all()
    assert_allclose(rows["contribution"], rows["weight"] * rows["estimate"])
    assert abs(r.reconstruction_error) < 1e-12
    used = d.filter(pl.col("time").count().over("id") == 9) if case == "complete" else d
    assert_allclose(r.twfe_estimate, dense_twfe(used), atol=2e-12)
    for row in rows.iter_rows(named=True):
        # Independently regress each actual two-block subset.
        if case == "future" or case == "two_period":
            normalized = used.with_columns(
                pl.when(pl.col("first_treat") > used["time"].max())
                .then(0)
                .otherwise(pl.col("first_treat"))
                .alias("first_treat")
            )
        else:
            normalized = used
        pair = normalized.filter(
            pl.col("first_treat").is_in([row["treated_cohort"], row["control_cohort"]])
            & pl.col("time").is_between(row["pre_start"], row["post_end"])
        )
        # Replace realized D with the comparison group's switch at the block boundary.
        pair = pair.with_columns(
            pl.when(pl.col("first_treat") == row["treated_cohort"])
            .then(row["post_start"])
            .otherwise(0)
            .alias("first_treat")
        )
        assert_allclose(row["estimate"], dense_twfe(pair), atol=4e-12)
    assert_allclose(r.summary()["weight"].sum(), 1)
    assert_allclose(r.summary()["contribution"].sum(), r.twfe_estimate, atol=2e-12)
    assert_allclose(
        r.diagnostics["already_treated_control_weight"],
        rows.filter(pl.col("control_already_treated"))["weight"].sum(),
    )


@pytest.mark.parametrize("seed", [1, 7, 83])
@pytest.mark.parametrize("periods", [3, 7, 12])
def test_dense_random_panels(seed, periods):
    d = simulate_panel(45, periods, seed=seed)
    r = bacon_decomp("y ~ 1", d, **KW)
    assert_allclose(r.twfe_estimate, dense_twfe(d), atol=3e-12)
    assert_allclose(r.decomposed_estimate, r.twfe_estimate, atol=3e-12)


def test_sources_partitions_identifiers_and_storage(tmp_path):
    d = simulate_panel(1300, 8, seed=215)
    a = bacon_decomp("y ~ 1", d, preprocess_mb=1, workdir=tmp_path / "work", **KW)
    d[:5000].write_parquet(tmp_path / "a.parquet", row_group_size=13)
    d[5000:].write_parquet(tmp_path / "b.parquet", row_group_size=31)
    b = bacon_decomp("y ~ 1", str(tmp_path / "*.parquet"), preprocess_mb=2, **KW)
    c = bacon_decomp(
        "y ~ 1",
        d.sample(fraction=1, shuffle=True, seed=6)
        .with_columns((pl.lit("u") + pl.col("id").cast(pl.String)).alias("id"))
        .lazy(),
        **KW,
    )
    for r in (b, c):
        assert_allclose(r.twfe_estimate, a.twfe_estimate, atol=1e-12)
        assert_allclose(r.comparisons()["estimate"], a.comparisons()["estimate"], atol=1e-12)
        assert_allclose(r.comparisons()["weight"], a.comparisons()["weight"], atol=1e-12)
    assert not list((tmp_path / "work").iterdir())
    assert a.diagnostics["partitions"] > 1
    report = json.loads(a.summary_json(tmp_path / "bacon.json", comparisons=True))
    assert len(report["comparisons"]) == a.comparisons().height
    assert json.loads((tmp_path / "bacon.json").read_text()) == report
    assert a.tidy().equals(a.comparisons())


def test_profile_and_guards(tmp_path, monkeypatch):
    from diff_stream import bacon

    d = fixture("heterogeneous")

    def no_decomposition(*args, **kwargs):
        raise AssertionError("allocated comparison table")

    monkeypatch.setattr(bacon, "_decompose", no_decomposition)
    p = bacon_decomp("y ~ 1", d, profile=True, **KW)
    assert p.notes["n_comparisons"] == 9
    assert p.notes["n_cohorts"] == 4
    with pytest.raises(MemoryError, match="preprocessing"):
        bacon_decomp("y ~ 1", d, memory_gb=0.01, workdir=tmp_path, **KW)
    assert not list(tmp_path.iterdir())
    # Finish validated staging, then refuse the result-table allocation.
    original_init = bacon.PanelStore.__init__

    def prepare(*args, **kwargs):
        original_init(*args, **kwargs)
        monkeypatch.setattr(bacon.MemoryBudget, "fits", lambda *args: False)

    monkeypatch.setattr(bacon.PanelStore, "__init__", prepare)
    with pytest.raises(MemoryError, match="comparison table"):
        bacon_decomp("y ~ 1", d, workdir=tmp_path, **KW)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "change,match",
    [
        ("incomplete", "balanced"),
        ("duplicate", "duplicate"),
        ("negative", "negative"),
        ("cohort_changes", "constant"),
        ("all_never", "two distinct"),
        ("one_timing", "two distinct"),
        ("missing", "missing"),
        ("nonconsecutive", "consecutive"),
    ],
)
def test_invalid_panels_cleanup(change, match, tmp_path):
    d = fixture("heterogeneous")
    if change == "incomplete":
        d = d.slice(1)
    elif change == "duplicate":
        d = pl.concat([d, d.head(1)])
    elif change == "always":
        d = d.with_columns(
            pl.when(pl.col("first_treat") == 2)
            .then(1)
            .otherwise(pl.col("first_treat"))
            .alias("first_treat")
        )
    elif change == "negative":
        d = d.with_columns(pl.lit(-1).alias("first_treat"))
    elif change == "cohort_changes":
        d = d.with_columns(
            pl.when((pl.col("id") == 0) & (pl.col("time") == 2))
            .then(5)
            .otherwise(pl.col("first_treat"))
            .alias("first_treat")
        )
    elif change == "all_never":
        d = d.filter(pl.col("first_treat") == 0)
    elif change == "one_timing":
        d = d.filter(pl.col("first_treat") == 5)
    elif change == "missing":
        d = d.with_columns(pl.lit(None, dtype=pl.Float64).alias("y"))
    elif change == "nonconsecutive":
        d = d.filter(pl.col("time") != 3)
    with pytest.raises(ValueError, match=match):
        bacon_decomp("y ~ 1", d, workdir=tmp_path, **KW)
    assert not list(tmp_path.iterdir())


def test_unsupported_options_and_strict_reference_requirement(monkeypatch):
    d = fixture("heterogeneous")
    original = PanelStore.__init__

    def no_scan(*args, **kwargs):
        raise AssertionError("executed rows")

    monkeypatch.setattr(PanelStore, "__init__", no_scan)
    with pytest.raises(ValueError, match="unbalanced"):
        bacon_decomp("y ~ 1", d, panel_mode="unbalanced", **KW)
    with pytest.raises(ValueError, match="numeric"):
        bacon_decomp("y ~ 1", d.with_columns(pl.lit("bad").alias("w")), weights="w", **KW)
    with pytest.raises(NotImplementedError):
        bacon_decomp("y ~ baseline | id", d, **KW)
    monkeypatch.setattr(PanelStore, "__init__", original)
    with pytest.raises(ValueError, match="never-treated"):
        cs_stream("y ~ 1", fixture("no_never"), reference_group="nevertreated", **KW)


def test_constant_effects_and_outcome_levels():
    for case, truth in [("constant", 3), ("null", 0)]:
        d = fixture(case)
        r = bacon_decomp("y ~ 1", d, **KW)
        assert_allclose(r.comparisons()["estimate"], truth, atol=3e-12)
        assert_allclose(r.twfe_estimate, truth, atol=3e-12)
    d = fixture("heterogeneous")
    a = bacon_decomp("y ~ 1", d, **KW)
    b = bacon_decomp(
        "y ~ 1",
        d.with_columns(pl.col("y") + 1e10 + pl.col("id") * 1e5 + pl.col("time") * 1e3),
        **KW,
    )
    assert_allclose(a.twfe_estimate, b.twfe_estimate, atol=2e-6)
    assert_allclose(a.comparisons()["weight"], b.comparisons()["weight"], atol=1e-14)
