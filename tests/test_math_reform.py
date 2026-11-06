"""Public math_reform worked-example replication against actual bacondecomp."""

import polars as pl
from numpy.testing import assert_allclose

from benchmarks._files import read_csv
from benchmarks.math_reform_replication import DATA, compare_replication, prepare_data
from diff_stream import bacon_decomp


def test_snapshot_and_treatment_mapping():
    raw = read_csv(DATA, null_values="NA")
    assert raw.columns == ["class", "state", "reform_math", "reformyr_math", "incearn_ln"]
    assert len(raw) == 520 and raw["state"].n_unique() == 26
    assert sorted(raw["class"].unique().to_list()) == list(range(1975, 1995))
    assert raw.select(pl.struct("state", "class").n_unique()).item() == 520
    assert raw.group_by("state").len()["len"].to_list() == [20] * 26
    assert raw["reformyr_math"].null_count() == 220
    assert (
        raw.select(pl.col("state", "class", "incearn_ln", "reform_math"))
        .null_count()
        .sum_horizontal()
        .item()
        == 0
    )
    d = prepare_data(raw)
    assert d.drop("first_treat").equals(raw)
    assert sorted(d["first_treat"].unique().to_list()) == [0, 1984, 1985, 1986, 1987]
    assert (d["reform_math"] == ((d["first_treat"] > 0) & (d["class"] >= d["first_treat"]))).all()
    assert d.filter(pl.col("first_treat") == 0)["state"].n_unique() == 11
    assert d.filter(pl.col("first_treat") > 0)["state"].n_unique() == 15


def test_exact_documented_example():
    fit, differences = compare_replication()
    assert fit.comparisons().height == 16
    assert fit.n_units == 26 and fit.n_obs == 520
    assert abs(fit.reconstruction_error) < 1e-12
    assert fit.diagnostics["dropped_units"] == 0 and fit.diagnostics["dropped_rows"] == 0
    assert max(differences.values()) < 1e-10
    assert_allclose(fit.twfe_estimate, 0.002459973, atol=5e-10)
    assert_allclose(fit.diagnostics["already_treated_control_weight"], 0.05265, atol=5e-6)


def test_layout_sources_and_id_relabeling(tmp_path):
    d = prepare_data()
    original, _ = compare_replication(d)
    shuffled = d.sample(fraction=1, shuffle=True, seed=193)
    mapping = {state: i for i, state in enumerate(sorted(d["state"].unique().to_list()))}
    relabeled = shuffled.with_columns(pl.col("state").replace_strict(mapping).alias("state"))
    relabeled.write_parquet(tmp_path / "math.parquet", row_group_size=13)
    for source in (shuffled.lazy(), str(tmp_path / "math.parquet")):
        fit = bacon_decomp(
            "incearn_ln ~ 1",
            source,
            panel=("state", "class"),
            cohort="first_treat",
            preprocess_mb=1,
            n_threads=2,
            workdir=tmp_path / "work",
        )
        assert_allclose(fit.twfe_estimate, original.twfe_estimate, atol=1e-12)
        assert_allclose(
            fit.comparisons()["estimate"], original.comparisons()["estimate"], atol=1e-12
        )
        assert_allclose(fit.comparisons()["weight"], original.comparisons()["weight"], atol=1e-14)
    assert not list((tmp_path / "work").iterdir())
