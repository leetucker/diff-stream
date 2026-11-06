"""Linear ETWFE: independent equations, reference commands and bounded fits."""

import json
import os
import sys

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from benchmarks._estimators import public
from benchmarks.etwfe_validation import (
    compare_reference,
    configuration,
    dense_oracle,
    sample,
    scenarios,
    verify_provenance,
)
from diff_stream import did_extended_twfe

SCENARIOS = list(scenarios())
KW = dict(panel=("id", "time"), cohort="first_treat", n_threads=2, batch_units=37, preprocess_mb=4)


def fit_case(panel, control, fe, case, A, inference):
    formula, weight = configuration(case)
    return did_extended_twfe(
        formula,
        sample(panel),
        **KW,
        control_group=public(
            "control_group", "nevertreated" if control == "never" else "notyettreated"
        ),
        fixed_effects=fe,
        treat_sample_as_fixed=public("treat_sample_as_fixed", inference),
        weights=weight,
        allow_time_varying_weights=True,
        incomplete_units=public(
            "incomplete_units", "unbalanced" if panel == "unbalanced" else "balanced"
        ),
        anticipation=A,
    )


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("inference", ["conditional", "unconditional"])
def test_independent_dense_stacked_inference(scenario, inference):
    panel, control, fe, case, A = scenario
    fit = fit_case(*scenario, inference)
    cells, beta, V, aggregates = dense_oracle(sample(panel), case, control, fe, A, inference)
    assert cells == fit.cells
    assert_allclose(fit.beta, beta, atol=2e-9, rtol=0)
    assert_allclose(fit.vcov, V, atol=2e-9, rtol=0)
    for kind, (b, v) in aggregates.items():
        a = fit.aggregate(public("kind", kind))
        assert_allclose(a.beta, b, atol=2e-9, rtol=0)
        assert_allclose(a.vcov, v, atol=2e-9, rtol=0)
    assert np.linalg.eigvalsh(fit.vcov).min() > -1e-9
    assert fit.n_obs == len(sample(panel))


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("inference", ["conditional", "unconditional"])
def test_authors_regression_and_marginal_effects(scenario, inference):
    fit = fit_case(*scenario, inference)
    prefix = "_".join(map(str, [*scenario, inference]))
    compare_reference(fit, prefix)


def test_reference_provenance():
    verify_provenance()


@pytest.mark.parametrize("control", ["nevertreated", "notyettreated"])
@pytest.mark.parametrize("fe", ["cohort", "unit"])
def test_weight_covariate_scaling_and_partition_order(control, fe, tmp_path):
    d = sample("unbalanced")
    kw = dict(
        **KW,
        incomplete_units="keep",
        weights="w",
        allow_time_varying_weights=True,
        control_group=public("control_group", control),
        fixed_effects=fe,
        workdir=tmp_path,
    )
    baseline = did_extended_twfe("y~x+z", d, **kw)
    source = d.with_columns(
        pl.col("w") * 1e8,
        pl.col("x") * 1e-7,
        ("unit_" + pl.col("id").cast(pl.String).str.pad_start(128, "0")).alias("id"),
    ).sample(fraction=1, shuffle=True, seed=43)
    options = {**kw, "batch_units": 19, "preprocess_mb": 1}
    actual = did_extended_twfe("y~x+z", source, **options)
    assert_allclose(actual.beta, baseline.beta, atol=2e-9, rtol=0)
    assert_allclose(actual.vcov, baseline.vcov, atol=2e-9, rtol=0)
    assert_allclose(
        actual.aggregate("overall").vcov, baseline.aggregate("overall").vcov, atol=2e-9, rtol=0
    )
    assert not list(tmp_path.iterdir())


def test_complete_profile_dispatch_and_reporting(tmp_path):
    d = sample("unbalanced")
    kw = dict(**KW, workdir=tmp_path, incomplete_units="drop")
    fit = did_extended_twfe("y~x", d, **kw)
    n = fit.n_units
    scaled = fit.with_vcov("CRV1")
    assert_allclose(scaled.vcov, fit.vcov * n / (n - 1))
    assert_allclose(scaled.aggregate("overall").vcov, fit.aggregate("overall").vcov * n / (n - 1))
    assert_allclose(scaled.with_vcov("CRV0").vcov, fit.vcov)
    assert fit.diagnostics["dropped_units"] > 0
    assert "cohort" in fit.cohort_time().columns
    assert fit.aggregate(balance_window=(-2, 1)).composition().height > 0
    assert fit.pretrend_test(horizons=[-2])["n_restrictions"] > 0
    assert json.loads(fit.summary_json())["depvar"] == "y"
    fit.cleanup()
    profile = did_extended_twfe("y~x", d, profile=True, **kw)
    assert profile.model == "diff-stream did_extended_twfe"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "option",
    [
        {"control_group": "bad"},
        {"fixed_effects": "bad"},
        {"treat_sample_as_fixed": "bad"},
        {"max_design_columns": 0},
        {"max_design_columns": True},
        {"batch_units": 0},
        {"n_threads": 0},
        {"incomplete_units": "bad"},
        {"anticipation": True},
        {"anticipation": -1},
        {"vcov": "iid"},
        {"allow_time_varying_weights": 1},
        {"preprocess_mb": 0},
    ],
)
def test_invalid_options_before_staging(option, tmp_path):
    with pytest.raises(ValueError):
        did_extended_twfe(
            "y~1",
            "missing.parquet",
            panel=("id", "time"),
            cohort="first_treat",
            workdir=tmp_path,
            **option,
        )
    assert not list(tmp_path.iterdir())


def test_identification_and_admission_cleanup(tmp_path):
    d = sample("balanced")
    kw = dict(**KW, workdir=tmp_path)
    with pytest.raises(ValueError, match="expanded design"):
        did_extended_twfe("y~x+z", d, max_design_columns=10, **kw)
    with pytest.raises(MemoryError):
        did_extended_twfe("y~1", d, max_memory_gb=0.01, **kw)
    with pytest.raises(ValueError, match="constant within unit"):
        did_extended_twfe("y~1", d, weights="w", **kw)
    # Remove an entire cohort's normalized reference date. The cohort/cell
    # intercept alias must not become an arbitrarily normalized treatment ATT.
    bad = d.filter(~((pl.col("first_treat") == 5) & (pl.col("time") == 4)))
    with pytest.raises(ValueError, match="unidentified"):
        did_extended_twfe("y~1", bad, incomplete_units="keep", **kw)
    with pytest.raises(TypeError, match="pretrend_leads"):
        did_extended_twfe("y~1", d, pretrend_leads=1, **kw)
    assert not list(tmp_path.iterdir())


@pytest.mark.resource
def test_etwfe_constrained_fit(resource_runner, resource_enforcement, tmp_path):
    from benchmarks.etwfe_validation import DATA

    path = DATA / "richer_unbalanced_panel.csv"
    work = tmp_path / "child"
    code = f"""import json,polars as pl
from diff_stream import did_extended_twfe
r=did_extended_twfe('y~x+z',pl.read_csv({str(path)!r}),panel=('id','time'),cohort='first_treat',control_group='not_yet_treated',fixed_effects='unit',incomplete_units='keep',weights='w',allow_time_varying_weights=True,n_threads=2,preprocess_mb=4,batch_units=37,workdir={str(work)!r})
a=r.aggregate('overall')
print(json.dumps(dict(beta=r.beta.tolist(),vcov=r.vcov.tolist(),att=a.beta.tolist(),att_vcov=a.vcov.tolist())))
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
    fit = fit_case("unbalanced", "notyet", "unit", "numeric_varying", 0, "unconditional")
    got = json.loads(result.stdout)
    a = fit.aggregate("overall")
    for key, expected in [
        ("beta", fit.beta),
        ("vcov", fit.vcov),
        ("att", a.beta),
        ("att_vcov", a.vcov),
    ]:
        assert_allclose(got[key], expected, atol=1e-9, rtol=0)
    assert not list(work.iterdir())


@pytest.mark.parametrize("fe", ["cohort", "unit"])
@pytest.mark.parametrize("inference", ["conditional", "unconditional"])
def test_categorical_compiler_matches_explicit_numeric_coding(fe, inference):
    d = sample("unbalanced")
    levels = sorted(d["category"].unique().to_list())
    coded = d.with_columns(
        (pl.col("category") == levels[1]).cast(pl.Float64).alias("ca"),
        (pl.col("category") == levels[2]).cast(pl.Float64).alias("cb"),
    )
    kw = dict(
        **KW,
        incomplete_units="keep",
        control_group="not_yet_treated",
        fixed_effects=fe,
        treat_sample_as_fixed=public("treat_sample_as_fixed", inference),
        weights="w",
        allow_time_varying_weights=True,
    )
    categorical = did_extended_twfe("y~x+C(category)", d, **kw)
    numeric = did_extended_twfe("y~x+ca+cb", coded, **kw)
    assert_allclose(categorical.beta, numeric.beta, atol=2e-9, rtol=0)
    assert_allclose(categorical.vcov, numeric.vcov, atol=2e-9, rtol=0)
    assert_allclose(
        categorical.aggregate("overall").vcov, numeric.aggregate("overall").vcov, atol=2e-9, rtol=0
    )
