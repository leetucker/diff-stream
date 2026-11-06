"""Clean last-treated windows: R references, sample policy and invariance."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from benchmarks._estimators import fit_estimator, public
from benchmarks._files import read_csv
from diff_stream import ReferenceWindowWarning, did_callaway_santanna, did_sun_abraham

DATA = Path(__file__).parent / "data"
REF = DATA / "reference_groups"
KW = dict(
    panel=("id", "time"),
    cohort="first_treat",
    n_threads=2,
    preprocess_mb=4,
    batch_units=37,
    effective_sample_warning=None,
)
# Sun-Abraham has no effective-sample warning (a CS option).
SA_KW = {k: v for k, v in KW.items() if k != "effective_sample_warning"}


def data(panel):
    return read_csv(
        DATA / ("richer_panel.csv" if panel == "balanced" else "richer_unbalanced_panel.csv")
    ).filter(pl.col("first_treat") != 0)


def compare_r(fit, prefix, kind=None):
    ref = read_csv(REF / (prefix + (".csv" if kind is None else "_" + kind + ".csv")))
    actual = (
        fit
        if kind is None
        else fit.aggregate(public("kind", "cohort" if kind == "group" else kind))
    )
    ix = [ref["term"].to_list().index(n) for n in actual.coefnames]
    V = read_csv(REF / (prefix + ("" if kind is None else "_" + kind) + "_vcov.csv")).to_numpy()
    if kind is None:
        native = read_csv(DATA / "reference_groups_native" / (prefix + ".csv"))
        ni = [native["term"].to_list().index(n) for n in actual.coefnames]
        assert_allclose(actual.beta, native["estimate"].to_numpy()[ni], atol=2e-7, rtol=0)
    assert_allclose(actual.beta, ref["estimate"].to_numpy()[ix], atol=2e-9, rtol=0)
    assert_allclose(actual.vcov, V[np.ix_(ix, ix)], atol=2e-9, rtol=0)


@pytest.mark.parametrize("panel", ["balanced", "unbalanced"])
@pytest.mark.parametrize("A", [0, 1])
@pytest.mark.parametrize("design", ["constant", "numeric"])
@pytest.mark.parametrize("control", ["nevertreated", "notyettreated"])
@pytest.mark.parametrize("base", ["varying", "universal"])
@pytest.mark.parametrize("method", ["reg", "ipw", "dr"])
def test_cs_last_reference_matches_r(panel, A, design, control, base, method):
    d = data(panel)
    with pytest.warns(ReferenceWindowWarning):
        fit = did_callaway_santanna(
            "y~1" if design == "constant" else "y~x+z",
            d,
            **KW,
            incomplete_units=public("incomplete_units", panel),
            anticipation=A,
            weights="w",
            allow_time_varying_weights=True,
            control_group=public("control_group", control),
            base_period=base,
            adjustment=public("adjustment", method),
        )
    prefix = f"cs_{panel}_{A}_{design}_{control}_{base}_{method}"
    compare_r(fit, prefix)
    for kind in ["dynamic", *(["simple", "group", "calendar"] if A == 0 else [])]:
        compare_r(fit, prefix, kind)
    assert fit.diagnostics["reference_cohort"] == 7
    assert fit.diagnostics["time_max"] == 6 - A
    assert all(g != 7 and t < 7 - A for g, t in fit.cells)


@pytest.mark.parametrize("panel", ["balanced", "unbalanced"])
@pytest.mark.parametrize("A", [0, 1])
@pytest.mark.parametrize("design", ["constant", "numeric", "extra_fe"])
def test_sa_last_reference_matches_fixest(panel, A, design):
    formula = "y~1" if design == "constant" else "y~x+z"
    if design == "extra_fe":
        formula += "|region^time"
    with pytest.warns(ReferenceWindowWarning):
        fit = did_sun_abraham(
            formula,
            data(panel),
            **SA_KW,
            incomplete_units=public("incomplete_units", panel),
            anticipation=A,
            weights="w",
            allow_time_varying_weights=True,
        )
    compare_r(fit, f"sunab_{panel}_{A}_{design}")


@pytest.mark.parametrize("estimator", ["cs", "sunab"])
@pytest.mark.parametrize("panel", ["balanced", "unbalanced", "complete"])
@pytest.mark.parametrize("A", [0, 1])
def test_explicit_restriction_invariance_and_cleanup(estimator, panel, A, tmp_path):
    d = data("balanced" if panel == "balanced" else "unbalanced")
    if panel == "complete":
        full = (
            d.group_by("id")
            .agg(pl.len().alias("n"))
            .filter(pl.col("n") == d["time"].n_unique())["id"]
        )
        expected = d.filter(pl.col("id").is_in(full.implode()))
    else:
        expected = d
    last = expected["first_treat"].max()
    expected = expected.filter(pl.col("time") < last - A).with_columns(
        pl.when(pl.col("first_treat") == last)
        .then(0)
        .otherwise(pl.col("first_treat"))
        .alias("first_treat")
    )
    kwargs = dict(
        **KW,
        estimator=estimator,
        incomplete_units=public("incomplete_units", panel),
        anticipation=A,
        weights="w",
        allow_time_varying_weights=True,
        workdir=tmp_path,
    )
    with pytest.warns(ReferenceWindowWarning):
        actual = fit_estimator("y~x+z", d, **kwargs)
    baseline = fit_estimator("y~x+z", expected, **kwargs)
    assert actual.cells == baseline.cells
    assert_allclose(actual.beta, baseline.beta, atol=2e-9, rtol=0)
    assert_allclose(actual.vcov, baseline.vcov, atol=2e-9, rtol=0)
    for kind in ["dynamic", "simple", "cohort", "calendar"]:
        assert_allclose(
            actual.aggregate(public("kind", kind)).beta,
            baseline.aggregate(public("kind", kind)).beta,
            atol=2e-9,
            rtol=0,
        )
        assert_allclose(
            actual.aggregate(public("kind", kind)).vcov,
            baseline.aggregate(public("kind", kind)).vcov,
            atol=2e-9,
            rtol=0,
        )
    assert actual.n_obs == len(expected)
    assert not list(tmp_path.iterdir())


def test_reference_choice_complete_drop_and_pooled_controls():
    d = read_csv(DATA / "richer_panel.csv")
    default = fit_estimator("y~1", d, **KW)
    strict = fit_estimator("y~1", d, last_treated_controls="never", **KW)
    assert_allclose(default.beta, strict.beta)
    assert default.diagnostics["reference"] == "never_treated"
    with pytest.warns(ReferenceWindowWarning):
        last = fit_estimator("y~1", d, last_treated_controls="always", **KW)
    assert last.diagnostics["reference_includes_never_treated"]
    assert last.diagnostics["reference_cohort"] == 7
    # Entire latest cohort loses a row: complete-panel selection must choose 6,
    # rather than choose 7 before discovering that it is absent from the fit.
    bad = data("balanced").filter(~((pl.col("first_treat") == 7) & (pl.col("time") == 2)))
    with pytest.warns(ReferenceWindowWarning):
        fit = fit_estimator("y~1", bad, incomplete_units="drop", **KW)
    assert fit.diagnostics["reference_cohort"] == 6
    assert {g for g, t in fit.cells} == {5}


def test_strict_reference_errors_and_profile(tmp_path):
    d = data("balanced")
    with pytest.raises(ValueError, match="never-treated"):
        fit_estimator("y~1", d, last_treated_controls="never", **KW)
    with pytest.raises(ValueError, match="two treated cohorts"):
        fit_estimator(
            "y~1",
            read_csv(DATA / "richer_panel.csv").filter(pl.col("first_treat").is_in([0, 7])),
            last_treated_controls="always",
            **KW,
        )
    with pytest.raises(ValueError, match="last_treated_controls"):
        fit_estimator("y~1", d, last_treated_controls="bad", **KW)
    with pytest.warns(ReferenceWindowWarning):
        p = fit_estimator("y~1", d, profile=True, workdir=tmp_path, **KW)
    assert p.notes["reference_cohort"] == 7
    assert not list(tmp_path.iterdir())


def test_external_reference_provenance():
    from benchmarks.reference_groups_validation import verify_provenance

    verify_provenance()
