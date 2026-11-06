"""Last-reference public APIs: native references, timing and sample invariance."""

import os
import sys
from pathlib import Path

import polars as pl
import pytest
from numpy.testing import assert_allclose

from benchmarks._estimators import fit_estimator, public
from benchmarks._files import read_csv
from benchmarks.etwfe_validation import dense_oracle as etwfe_dense
from benchmarks.reference_population_review import KW, cases, configurations, sample
from benchmarks.reference_population_validation import compare, options, provenance
from diff_stream import (
    ReferenceWindowWarning,
    did_bjs,
    did_extended_twfe,
    did_two_stage,
)

CASES = list(cases())
ENTRIES = [("bjs", did_bjs), ("gardner", did_two_stage), ("etwfe", did_extended_twfe)]


def extra(name):
    return {}


@pytest.mark.parametrize("c", CASES, ids=lambda c: c["name"])
@pytest.mark.parametrize(
    "name", ["bjs", "gardner_static", "gardner_dynamic", "etwfe_conditional", "etwfe_unconditional"]
)
def test_native_reference_and_explicit_restriction(c, name):
    _, kwargs, entry = next(x for x in configurations(c) if x[0] == name)
    data = sample(c)
    with pytest.warns(ReferenceWindowWarning):
        result = entry(data=data, **kwargs, **options(c), **extra(name))
    try:
        compare(result.aggregate("event_time") if name == "bjs" else result, c["name"] + "_" + name)
        with pytest.warns(ReferenceWindowWarning):
            explicit = entry(
                data=data, last_treated_controls="always", **kwargs, **options(c), **extra(name)
            )
        baseline = entry(
            data=sample(c, True),
            last_treated_controls="never",
            **kwargs,
            **options(c),
            **extra(name),
        )
        try:
            assert_allclose(result.beta, baseline.beta, atol=2e-9, rtol=0)
            assert_allclose(result.vcov, baseline.vcov, atol=2e-9, rtol=0)
            assert_allclose(result.beta, explicit.beta, atol=2e-9, rtol=0)
            assert result.diagnostics["reference_cohort"] == 7
            assert result.diagnostics["time_max"] == 6 - c["A"]
            assert result.diagnostics["reference_cutoff"] == 7 - c["A"]
            if name.startswith("gardner"):
                assert 7 not in result.support()["cohort"]
                for horizon in result.coefnames:
                    assert horizon in baseline.coefnames
            else:
                assert all(g != 7 and t < 7 - c["A"] for g, t in result.cells)
                for kind in ["dynamic", "simple", "cohort", "calendar"]:
                    assert_allclose(
                        result.aggregate(public("kind", kind)).beta,
                        baseline.aggregate(public("kind", kind)).beta,
                        atol=2e-9,
                        rtol=0,
                    )
                    assert_allclose(
                        result.aggregate(public("kind", kind)).vcov,
                        baseline.aggregate(public("kind", kind)).vcov,
                        atol=2e-9,
                        rtol=0,
                    )
        finally:
            explicit.cleanup()
            baseline.cleanup()
    finally:
        result.cleanup()


@pytest.mark.parametrize("c", CASES, ids=lambda c: c["name"])
def test_bjs_original_onset_placebos_native_and_restricted(c, tmp_path):
    fml = "y~x+I(x**2)|region^time" if c["rich"] else "y~1"
    full = sample(c)
    raw = full.filter(pl.col("time") < 7 - c["A"])
    with pytest.warns(ReferenceWindowWarning):
        result = did_bjs(
            fml,
            full,
            pretrend_leads=2,
            workdir=tmp_path,
            **options(c),
        )
    baseline = did_bjs(fml, raw, pretrend_leads=2, workdir=tmp_path, **options(c))
    try:
        compare(result.placebos(), c["name"] + "_bjs_placebos")
        assert_allclose(result.placebos().beta, baseline.placebos().beta, atol=2e-9, rtol=0)
        assert_allclose(result.placebos().vcov, baseline.placebos().vcov, atol=2e-9, rtol=0)
        assert 7 in result.placebos().support()["cohort"]
        assert result.diagnostics["placebo_cohort_policy"].startswith("original known")
    finally:
        result.cleanup()
        baseline.cleanup()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("control", ["nevertreated", "notyettreated"])
@pytest.mark.parametrize("fe", ["cohort", "unit"])
@pytest.mark.parametrize("inference", ["conditional", "unconditional"])
@pytest.mark.parametrize("A", [0, 1])
def test_etwfe_both_control_and_fe_conventions(control, fe, inference, A):
    c = dict(panel="unbalanced", A=A, rich=True)
    with pytest.warns(ReferenceWindowWarning):
        result = did_extended_twfe(
            "y~x+z",
            sample(c),
            control_group=public("control_group", control),
            fixed_effects=fe,
            treat_sample_as_fixed=public("treat_sample_as_fixed", inference),
            **options(c),
        )
    try:
        cells, b, V, aggregates = etwfe_dense(
            sample(c, True),
            "numeric_varying",
            "never" if control == "nevertreated" else "notyet",
            fe,
            A,
            inference,
        )
        assert result.cells == cells
        assert_allclose(result.beta, b, atol=2e-9, rtol=0)
        assert_allclose(result.vcov, V, atol=2e-9, rtol=0)
        for kind, (bb, vv) in aggregates.items():
            agg = result.aggregate(public("kind", kind))
            assert_allclose(agg.beta, bb, atol=2e-9, rtol=0)
            assert_allclose(agg.vcov, vv, atol=2e-9, rtol=0)
    finally:
        result.cleanup()


@pytest.mark.parametrize("name,entry", ENTRIES)
@pytest.mark.parametrize("A", [0, 1])
@pytest.mark.parametrize("mode", ["balanced", "unbalanced", "complete"])
def test_explicit_last_pools_existing_controls_and_cleanup(name, entry, A, mode, tmp_path):
    data = read_csv(
        Path(__file__).parent
        / "data"
        / ("richer_panel.csv" if mode == "balanced" else "richer_unbalanced_panel.csv")
    )
    retained = (
        data.filter(pl.len().over("id") == data["time"].n_unique()) if mode == "complete" else data
    )
    canonical = retained.filter(pl.col("time") < 7 - A).with_columns(
        pl.when(pl.col("first_treat") == 7)
        .then(0)
        .otherwise(pl.col("first_treat"))
        .alias("first_treat")
    )
    opt = dict(
        **KW,
        weights="w",
        allow_time_varying_weights=True,
        incomplete_units=public("incomplete_units", mode),
        anticipation=A,
        workdir=tmp_path,
        **extra(name),
    )
    with pytest.warns(ReferenceWindowWarning):
        result = entry("y~x+z", data, last_treated_controls="always", **opt)
    baseline = entry("y~x+z", canonical, **opt)
    try:
        assert result.diagnostics["reference_includes_never_treated"]
        assert_allclose(result.beta, baseline.beta, atol=2e-9, rtol=0)
        assert_allclose(result.vcov, baseline.vcov, atol=2e-9, rtol=0)
        assert result.n_obs == len(canonical)
    finally:
        result.cleanup()
        baseline.cleanup()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("name,entry", ENTRIES)
def test_complete_selection_precedes_reference_choice(name, entry):
    data = sample(CASES[0]).filter(~((pl.col("first_treat") == 7) & (pl.col("time") == 7)))
    with pytest.warns(ReferenceWindowWarning):
        result = entry("y~1", data, incomplete_units="drop", **KW, **extra(name))
    try:
        assert result.diagnostics["reference_cohort"] == 6
        assert result.diagnostics["time_max"] == 5
        assert (
            result.diagnostics.get(
                "dropped_units", result.diagnostics.get("input_units", 0) - result.n_units
            )
            > 0
        )
    finally:
        result.cleanup()


@pytest.mark.parametrize("name,entry", ENTRIES)
def test_policy_guards_profiles_and_default_invariance(name, entry, tmp_path):
    data = sample(CASES[0])
    with pytest.raises(ValueError, match="never-treated"):
        entry("y~1", data, last_treated_controls="never", **KW, **extra(name))
    with pytest.raises(ValueError, match="last_treated_controls"):
        entry("y~1", data, last_treated_controls="bad", workdir=tmp_path, **KW, **extra(name))
    with pytest.warns(ReferenceWindowWarning):
        prof = entry("y~1", data, profile=True, workdir=tmp_path, **KW, **extra(name))
    assert prof.notes["reference_cohort"] == 7
    assert not list(tmp_path.iterdir())
    never = read_csv(Path(__file__).parent / "data/richer_panel.csv")
    default = entry("y~1", never, **KW, **extra(name))
    strict = entry("y~1", never, last_treated_controls="never", **KW, **extra(name))
    try:
        assert_allclose(default.beta, strict.beta, atol=1e-12, rtol=0)
        assert_allclose(default.vcov, strict.vcov, atol=1e-12, rtol=0)
        assert default.diagnostics["reference"] == "never_treated"
    finally:
        default.cleanup()
        strict.cleanup()


@pytest.mark.parametrize("name,entry", ENTRIES)
def test_dispatch_and_units_disappearing_from_window(name, entry, tmp_path):
    data = sample(CASES[0])
    tail = data.head(1).with_columns(
        pl.lit(99999).cast(data.schema["id"]).alias("id"),
        pl.lit(7).cast(data.schema["time"]).alias("time"),
    )
    data = pl.concat([data, tail])
    opt = dict(**KW, incomplete_units="keep", workdir=tmp_path, **extra(name))
    with pytest.warns(ReferenceWindowWarning):
        direct = entry("y~1", data, last_treated_controls="always", **opt)
    with pytest.warns(ReferenceWindowWarning):
        dispatch = fit_estimator("y~1", data, estimator=name, last_treated_controls="always", **opt)
    try:
        assert direct.diagnostics["reference_dropped_units"] == 1
        assert_allclose(direct.beta, dispatch.beta, atol=1e-12, rtol=0)
        assert_allclose(direct.vcov, dispatch.vcov, atol=1e-12, rtol=0)
    finally:
        direct.cleanup()
        dispatch.cleanup()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("name,entry", ENTRIES)
def test_insufficient_clean_date_support_is_not_extrapolated(name, entry, tmp_path):
    data = sample(CASES[0]).filter(~((pl.col("first_treat") == 7) & (pl.col("time") == 6)))
    with (
        pytest.warns(ReferenceWindowWarning),
        pytest.raises(ValueError, match="unidentified|not identified|singular nuisance"),
    ):
        entry("y~1", data, incomplete_units="keep", workdir=tmp_path, **KW, **extra(name))
    assert not list(tmp_path.iterdir())


def test_native_provenance():
    provenance()


@pytest.mark.resource
def test_new_reference_windows_constrained(resource_runner, resource_enforcement, tmp_path):
    path = Path(__file__).parent / "data/richer_unbalanced_panel.csv"
    code = f"""from diff_stream import did_bjs,did_two_stage,did_extended_twfe
import polars as pl,json
d=pl.read_csv({str(path)!r}).filter(pl.col('first_treat')!=0)
out=[]
for entry in [did_bjs,did_two_stage,did_extended_twfe]:
 kw={{'pretrend_leads':2}} if entry is did_bjs else {{}}
 with entry('y~x+z',d,panel=('id','time'),cohort='first_treat',last_treated_controls='always',incomplete_units='keep',anticipation=1,weights='w',allow_time_varying_weights=True,n_threads=2,batch_units=37,preprocess_mb=4,workdir={str(tmp_path / "child")!r},**kw) as r:
  assert r.diagnostics['reference_cutoff']==6
  out.append(r.beta.tolist())
print(json.dumps(out))
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
    assert not list((tmp_path / "child").iterdir())
