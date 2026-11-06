"""Conditional CS replication, nuisance-score and bounded execution checks."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from benchmarks._estimators import public
from benchmarks._files import read_csv
from diff_stream import did_callaway_santanna
from diff_stream.simulate import simulate_conditional_panel

ROOT = Path(__file__).parent / "data"
KW = dict(panel=("id", "time"), cohort="first_treat", n_threads=2)


@pytest.fixture(scope="module")
def conditional():
    return read_csv(ROOT / "conditional_panel.csv")


@pytest.mark.parametrize("formula", ["numeric", "expanded"])
@pytest.mark.parametrize("control", ["nevertreated", "notyettreated"])
@pytest.mark.parametrize("base", ["varying", "universal"])
@pytest.mark.parametrize("method", ["reg", "ipw", "dr"])
def test_r_did_conditional(conditional, formula, control, base, method):
    fml = "y ~ x + z" if formula == "numeric" else "y ~ x*C(category) + I(x**2) + z"
    prefix = f"cs_{formula}_{control}_{base}_{method}"
    ref = read_csv(ROOT / "cs_reference" / f"{prefix}.csv", null_values="NA")
    r = did_callaway_santanna(
        fml,
        conditional,
        control_group=public("control_group", control),
        base_period=base,
        adjustment=public("adjustment", method),
        **KW,
    )
    keys = list(zip(ref["g"], ref["t"], strict=True))
    ix = [keys.index(c) for c in r.cells]
    assert_allclose(r.beta, ref["att"].to_numpy()[ix], atol=2e-8)
    V = read_csv(ROOT / "cs_reference" / f"{prefix}_vcov.csv").to_numpy()
    assert_allclose(r.vcov, V[np.ix_(ix, ix)], atol=2e-8)
    for kind in ["dynamic", "simple"]:
        a = r.aggregate(public("kind", kind))
        refa = read_csv(ROOT / "cs_reference" / f"{prefix}_{kind}.csv", null_values="NA")
        if kind == "dynamic":
            levels = [int(n.split("::")[-1]) for n in a.coefnames]
            refa = refa.filter(pl.col("event").is_in(levels)).sort("event")
        assert_allclose(a.beta, refa["att"], atol=2e-8)
        assert_allclose(a.se, refa["se"], atol=2e-8)


def test_partition_batch_and_crv_invariance(conditional):
    f = "y ~ x + z"
    a = did_callaway_santanna(
        f, conditional, adjustment="doubly_robust", batch_units=43, preprocess_mb=1, **KW
    )
    b = did_callaway_santanna(
        f,
        conditional.sample(fraction=1, shuffle=True, seed=11),
        adjustment="doubly_robust",
        vcov="CRV1",
        **KW,
    )
    factor = a.n_units / (a.n_units - 1)
    assert_allclose(a.beta, b.beta, atol=1e-9)
    assert_allclose(a.vcov * factor, b.vcov, atol=1e-9)
    assert_allclose(a.with_vcov("CRV1").aggregate().vcov, b.aggregate().vcov, atol=1e-9)


@pytest.mark.parametrize("method", ["ipw", "dr"])
@pytest.mark.parametrize("mode", ["balanced", "unbalanced"])
@pytest.mark.parametrize("scale", [1e-12, 1e12])
def test_covariate_units_preserve_joint_estimates_and_covariance(conditional, method, mode, scale):
    d = conditional
    if mode == "unbalanced":
        d = d.filter(~((pl.col("id") % 4 == 0) & pl.col("time").is_in([2, 4])))
    fits = []
    try:
        for data in (d, d.with_columns(pl.col("x") * scale)):
            fits.append(
                did_callaway_santanna(
                    "y ~ x + z",
                    data,
                    adjustment=public("adjustment", method),
                    incomplete_units=public("incomplete_units", mode),
                    batch_units=137,
                    **KW,
                )
            )
        assert fits[0].cells == fits[1].cells
        assert_allclose(fits[0].beta, fits[1].beta, atol=1e-9, rtol=0)
        assert_allclose(fits[0].vcov, fits[1].vcov, atol=1e-9, rtol=0)
        assert_allclose(
            fits[0].aggregate("overall").vcov, fits[1].aggregate("overall").vcov, atol=1e-9, rtol=0
        )
    finally:
        for fit in fits:
            fit.cleanup()


def test_reuses_only_identical_nuisance_inputs():
    constant = simulate_conditional_panel(400, 5, time_varying=False)
    varying = simulate_conditional_panel(400, 5, time_varying=True)
    a = did_callaway_santanna("y ~ x + z", constant, **KW)
    b = did_callaway_santanna("y ~ x + z", varying, **KW)
    assert a.diagnostics["covariates_time_invariant"]
    assert a.diagnostics["propensity_models"] == 3
    changing_controls = did_callaway_santanna(
        "y ~ x + z", constant, control_group="not_yet_treated", **KW
    )
    assert changing_controls.diagnostics["propensity_models"] > a.diagnostics["propensity_models"]
    assert not b.diagnostics["covariates_time_invariant"]
    assert b.diagnostics["propensity_models"] > a.diagnostics["propensity_models"]
    # Propensity models use shared data passes, rather than one pass per model.
    assert b.diagnostics["nuisance_source_passes"] < 25


@pytest.mark.parametrize(
    "fml,match",
    [
        ("y ~ missing", "unknown variables"),
        ("y ~ x + I(x)", "singular"),
        ("y ~ log(-abs(x)-1)", "non-finite"),
        ("y ~ x | id", "supports"),
    ],
)
def test_design_errors_and_cleanup(conditional, tmp_path, fml, match):
    with pytest.raises((ValueError, NotImplementedError), match=match):
        did_callaway_santanna(fml, conditional, workdir=tmp_path, **KW)
    assert list(tmp_path.iterdir()) == []


def test_missing_covariate_is_not_silently_dropped(conditional):
    d = conditional.with_columns(
        pl.when(pl.col("id") == 0).then(None).otherwise(pl.col("x")).alias("x")
    )
    with pytest.raises(ValueError, match="missing"):
        did_callaway_santanna("y ~ x", d, **KW)


def test_large_category_rejected_in_bounded_discovery(conditional):
    d = conditional.with_columns(pl.col("id").cast(pl.String).alias("category"))
    with pytest.raises(ValueError, match="256"):
        did_callaway_santanna("y ~ C(category)", d, **KW)


def test_perfect_separation_rejected(conditional):
    d = conditional.with_columns((pl.col("first_treat") > 0).cast(pl.Float64).alias("separator"))
    with pytest.raises(ValueError, match="separation|singular"):
        did_callaway_santanna("y ~ separator", d, adjustment="ipw", **KW)


@pytest.mark.parametrize(
    "kwargs", [{"adjustment": "invalid"}, {"propensity_trim": 0}, {"propensity_trim": 1.1}]
)
def test_option_errors(conditional, kwargs):
    with pytest.raises(ValueError):
        did_callaway_santanna("y ~ x", conditional, **KW, **kwargs)


@pytest.mark.parametrize("method", ["reg", "ipw", "dr"])
@pytest.mark.parametrize("level", [0.8, 0.995, 1])
def test_control_trimming_matches_r_dr_did(method, level):
    data = read_csv(ROOT / "trim_panel.csv")
    r = did_callaway_santanna(
        "y ~ x", data, adjustment=public("adjustment", method), propensity_trim=level, **KW
    )
    ref = read_csv(ROOT / "cs_reference" / f"trim_{method}_{level:g}.csv")
    assert_allclose(r.beta, ref["att"], atol=2e-8)
    assert_allclose(r.se, ref["se"], atol=2e-8)
    diag = r.diagnostics["comparisons"][0]
    assert (
        diag["n_treated"] == data.filter((pl.col("time") == 1) & (pl.col("first_treat") > 0)).height
    )
    if method != "reg" and level == 0.8:
        assert diag["trimmed_controls"] > 0


@pytest.mark.parametrize("method", ["reg", "dr"])
def test_exact_linear_control_trend(method):
    data = read_csv(ROOT / "trim_panel.csv")
    data = data.with_columns(
        (
            pl.col("id") * 0.01
            + pl.col("time") * (0.4 + 0.7 * pl.col("x"))
            + ((pl.col("first_treat") > 0) & (pl.col("time") == 2)).cast(pl.Float64)
        ).alias("y")
    )
    r = did_callaway_santanna("y ~ x", data, adjustment=public("adjustment", method), **KW)
    assert_allclose(r.beta, [1], atol=1e-12)
    assert r.se[0] < 1e-12


def test_i_and_numeric_transforms_use_family_compiler(conditional):
    a = did_callaway_santanna(
        "y ~ x + i(category) + np.log(I(x*x+1))", conditional, adjustment="regression", **KW
    )
    b = did_callaway_santanna(
        "y ~ x + C(category) + log(I(x*x+1))", conditional, adjustment="regression", **KW
    )
    assert_allclose(a.beta, b.beta, atol=1e-10)
    assert_allclose(a.vcov, b.vcov, atol=1e-10)


@pytest.mark.parametrize("method", ["reg", "ipw", "dr"])
def test_intercept_fast_path_matches_fitted_constant(conditional, method):
    a = did_callaway_santanna("y ~ 1", conditional, adjustment=public("adjustment", method), **KW)
    b = did_callaway_santanna(
        "y ~ I(1) - 1", conditional, adjustment=public("adjustment", method), **KW
    )
    assert_allclose(a.beta, b.beta, atol=1e-10)
    assert_allclose(a.vcov, b.vcov, atol=1e-10)
    if method != "reg":
        with pytest.raises(ValueError, match="no controls"):
            did_callaway_santanna(
                "y ~ 1",
                conditional,
                adjustment=public("adjustment", method),
                propensity_trim=0.01,
                **KW,
            )
