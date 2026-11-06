"""Clean pre-trend selection and CS unit-multiplier inference."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from hdfe_stream.core.wildboot import WEIGHTS
from numpy.testing import assert_allclose
from scipy.special import ndtri
from scipy.stats import chi2
from stream_limits import current_rss_gb

from benchmarks._estimators import fit_estimator, public
from benchmarks._files import read_csv
from diff_stream import did_bjs
from diff_stream._core.multiplier import StreamingMultiplier, multiplier_band
from diff_stream.store import PanelStore

ROOT = Path(__file__).parent / "data"
REF = ROOT / "inference_reference"
KW = dict(panel=("id", "time"), cohort="first_treat", n_threads=2)
REPS = 199
SCENARIOS = [
    ("constant", "dr", "nevertreated", "varying"),
    ("numeric", "reg", "nevertreated", "varying"),
    ("numeric", "ipw", "notyettreated", "universal"),
    *[
        ("numeric", "dr", control, base)
        for control in ("nevertreated", "notyettreated")
        for base in ("varying", "universal")
    ],
]


@pytest.fixture(scope="module")
def panel():
    return read_csv(ROOT / "anticipation_panel.csv").with_columns(
        (0.2 + pl.col("id") % 9).alias("w")
    )


@pytest.fixture
def common_draws(panel, monkeypatch):
    """Align stored common multipliers by original ID, solely for R validation.

    Production assigns RNG draws by staged traversal without a global ID table.
    This small test fixture maps unique baseline outcomes to IDs, then replaces
    only the sampler; it exercises the actual streaming score reducer.
    """
    xi = read_csv(REF / "multipliers.csv.gz", has_header=False).to_numpy()
    baseline = panel.filter(pl.col("time") == 1)
    lookup = dict(zip(baseline["y"], baseline["id"], strict=True))
    active = {}
    original_batches = PanelStore.batches
    original_init = StreamingMultiplier.__init__

    def batches(self, **kwargs):
        for raw in original_batches(self, **kwargs):
            active["ids"] = np.array([lookup[y] for y in raw[1][:, 0]])
            active["position"] = 0
            yield raw

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        def sample(n, rng):
            start = active["position"]
            stop = start + n // self.reps
            active["position"] = stop
            return xi[active["ids"][start:stop]].ravel()

        self.sampler = sample

    monkeypatch.setattr(PanelStore, "batches", batches)
    monkeypatch.setattr(StreamingMultiplier, "__init__", init)
    return xi


@pytest.mark.parametrize("unbalanced", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("A", [0, 2])
@pytest.mark.parametrize("design,method,control,base", SCENARIOS)
def test_r_common_draws(
    panel, common_draws, unbalanced, weighted, A, design, method, control, base
):
    d = (
        panel.filter(~((pl.col("id") % 4 == 0) & pl.col("time").is_in([3, 7])))
        if unbalanced
        else panel
    )
    sample = "unbalanced" if unbalanced else "balanced"
    r = fit_estimator(
        "y ~ 1" if design == "constant" else "y ~ x + z",
        d,
        weights="w" if weighted else None,
        incomplete_units=public("incomplete_units", sample),
        anticipation=A,
        adjustment=public("adjustment", method),
        control_group=public("control_group", control),
        base_period=base,
        multiplier_bootstrap_reps=REPS,
        multiplier_bootstrap_seed=5,
        batch_units=17,
        **KW,
    )
    prefix = f"{sample}_{int(weighted)}_{A}_{design}_{control}_{base}_{method}"
    for kind in ("cells", "dynamic", "simple", "cohort", "calendar"):
        a = r if kind == "cells" else r.aggregate(public("kind", kind))
        label = "group" if kind == "cohort" else kind
        ref = read_csv(REF / f"{prefix}_{label}.csv.gz")
        ix = [ref["term"].to_list().index(n) for n in a.coefnames]
        draws = read_csv(REF / f"{prefix}_{label}_draws.csv.gz").to_numpy()[:, ix]
        assert_allclose(a.multiplier_draws(), draws, atol=2e-8)
        band = a.multiplier_confint()
        for col, rc in [
            ("Estimate", "estimate"),
            ("Std. Error", "se"),
            ("critical_value", "critical"),
            ("2.5%", "lower"),
            ("97.5%", "upper"),
        ]:
            assert_allclose(band[col], ref[rc].to_numpy()[ix], atol=2e-8)
        assert a.multiplier_confint(keep=a.coefnames[:1], exact_match=True).height == 1
    p = r.pretrend_test()
    reference = read_csv(REF / "pretests.csv").filter(pl.col("case") == prefix).row(0, named=True)
    assert_allclose(p["statistic"], reference["statistic"], atol=2e-7)
    assert_allclose(p["pvalue"], reference["pvalue"], atol=1e-8)
    assert p["df1"] == reference["df"]
    assert all(h < -A for h in p["horizons"])
    direct, aggregated = r.pretrend_test(level="event_time"), r.aggregate().pretrend_test()
    for key in ("statistic", "pvalue"):
        assert_allclose(direct.pop(key), aggregated.pop(key), atol=1e-12)
    assert direct == aggregated


@pytest.mark.parametrize("weights", list(WEIGHTS))
def test_reducer_reuses_hdfe_samplers_and_bounds_blocks(weights, monkeypatch):
    from diff_stream._core import multiplier

    scores = np.random.default_rng(12).normal(size=(53, 5))
    xi = WEIGHTS[weights](53 * 107, np.random.default_rng(42)).reshape(53, 107)
    monkeypatch.setattr(multiplier, "MAX_DRAW_ENTRIES", 107 * 7)
    a = StreamingMultiplier(5, 107, 42, weights)
    a.update(scores[:21, :2], scores[:21, 2:])
    a.update(scores[21:, :2], scores[21:, 2:])
    assert_allclose(a.draws, xi.T @ scores, atol=2e-12)
    assert a.n_clusters == 53
    b = StreamingMultiplier(5, 107, 42, weights)
    b.update(scores)
    assert_allclose(a.draws, b.draws, atol=2e-12)


@pytest.mark.parametrize("estimator", ["cs", "sunab"])
@pytest.mark.parametrize("base", ["varying", "universal"])
def test_clean_pretrend_selection(panel, estimator, base):
    # Sun-Abraham always omits the last clean period; base_period is a CS option.
    base = {"base_period": base} if estimator == "cs" else {}
    r = fit_estimator("y ~ 1", panel, estimator=estimator, anticipation=2, **base, **KW)
    test = r.pretrend_test(horizons=[-4, -5])
    ix = [i for i, (g, t) in enumerate(r.cells) if t - g in (-4, -5)]
    b, V = r.beta[ix], r.vcov[np.ix_(ix, ix)]
    W = b @ np.linalg.solve(V, b)
    assert_allclose(test["statistic"], W)
    assert_allclose(test["pvalue"], chi2.sf(W, len(ix)))
    assert test["horizons"] == [-5, -4]
    assert test["reference_event_time"] == -3
    assert len(test["cells"]) == len(ix)
    for bad in [[-2], [-1], [0], [-99], [], [True], [1.2]]:
        with pytest.raises(ValueError, match="clean|not estimated"):
            r.pretrend_test(horizons=bad)
    with pytest.raises(ValueError, match="event-time"):
        r.aggregate("overall").pretrend_test()
    with pytest.raises(ValueError, match="level"):
        r.pretrend_test(level="invalid")


def test_unsupported_and_degenerate_pretests(panel):
    b = did_bjs("y ~ 1", panel, anticipation=2, **KW)
    with pytest.raises(ValueError, match="pretrend_leads"):
        b.pretrend_test()
    b.cleanup()
    f = fit_estimator("y ~ 1", panel, **KW)
    f.vcov[:] = 0
    with pytest.raises(ValueError, match="zero"):
        f.pretrend_test()
    f.vcov[:] = 1
    with pytest.raises(ValueError, match="singular"):
        f.pretrend_test()
    with pytest.raises(ValueError, match="no estimated"):
        fit_estimator("y ~ 1", panel, **KW).aggregate(horizons=[0]).pretrend_test()


def test_band_quantiles_selection_crv_and_no_refit(panel, tmp_path, monkeypatch):
    f = fit_estimator(
        "y ~ x + z",
        panel,
        weights="w",
        multiplier_bootstrap_reps=299,
        multiplier_bootstrap_seed=37,
        workdir=tmp_path,
        **KW,
    )
    g = fit_estimator(
        "y ~ x + z",
        panel,
        weights="w",
        multiplier_bootstrap_reps=299,
        multiplier_bootstrap_seed=37,
        batch_units=11,
        **KW,
    )
    assert_allclose(f.multiplier_draws(), g.multiplier_draws(), atol=1e-11)
    assert not list(tmp_path.iterdir())
    monkeypatch.setattr(
        PanelStore, "batches", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("reread"))
    )
    a = f.aggregate(balance_window=(-3, 1))
    keep = a.coefnames[:2]
    draws = a.multiplier_draws(keep=keep, exact_match=True)
    q = np.sort(draws, axis=0)
    # Explicit nearest-rank quantiles, independently of multiplier_band.
    scale = (q[int(np.ceil(0.75 * 299)) - 1] - q[int(np.ceil(0.25 * 299)) - 1]) / (
        ndtri(0.75) - ndtri(0.25)
    )
    critical = np.sort(np.max(np.abs(draws / scale), axis=1))[int(np.ceil(0.95 * 299)) - 1]
    band = a.multiplier_confint(keep=keep, exact_match=True)
    assert_allclose(band["Std. Error"], scale)
    assert_allclose(band["critical_value"], critical)
    point = a.multiplier_confint(keep=keep, exact_match=True, joint=False)
    assert np.all(point["critical_value"].to_numpy() <= critical)
    assert_allclose(
        a.with_vcov("CRV1").multiplier_draws(), a.multiplier_draws() * np.sqrt(400 / 399)
    )
    assert_allclose(
        f.with_vcov("CRV1").aggregate().multiplier_draws(),
        f.aggregate().multiplier_draws() * np.sqrt(400 / 399),
    )
    analytic = a.multiplier_confint(studentization="analytic")
    assert_allclose(analytic["Std. Error"], a.se)
    copy = a.multiplier_draws()
    copy[:] = 0
    assert np.any(a.multiplier_draws())


@pytest.mark.parametrize(
    "option,value",
    [
        ("multiplier_bootstrap_reps", True),
        ("multiplier_bootstrap_reps", 1),
        ("multiplier_bootstrap_reps", -1),
        ("multiplier_bootstrap_reps", 1.5),
        ("multiplier_bootstrap_seed", True),
        ("multiplier_bootstrap_seed", -1),
        ("multiplier_bootstrap_seed", 1.5),
        ("multiplier_bootstrap_weights", "bad"),
    ],
)
def test_invalid_options_before_execution(panel, option, value, monkeypatch):
    monkeypatch.setattr(
        PanelStore, "__init__", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("executed"))
    )
    with pytest.raises(ValueError, match="bootstrap"):
        fit_estimator("y ~ 1", panel, **{option: value}, **KW)


def test_disabled_profile_and_invalid_inference(panel):
    f = fit_estimator("y ~ 1", panel, **KW)
    with pytest.raises(ValueError, match="bootstrap_reps"):
        f.multiplier_confint()
    for estimator in ("sunab", "bjs"):
        with pytest.raises(TypeError, match="multiplier_bootstrap_reps"):
            fit_estimator("y ~ 1", panel, estimator=estimator, multiplier_bootstrap_reps=99, **KW)
    p = fit_estimator("y ~ 1", panel, multiplier_bootstrap_reps=999, profile=True, **KW)
    assert p.notes["multiplier_bootstrap_reps"] == 999
    with pytest.raises(ValueError, match="zero"):
        multiplier_band(np.zeros((99, 2)), 0.05)
    for alpha in [0, 1, -0.1, np.nan, True]:
        with pytest.raises(ValueError, match="alpha"):
            multiplier_band(np.ones((99, 2)), alpha)
    with pytest.raises(ValueError, match="joint"):
        multiplier_band(np.ones((99, 2)), 0.05, joint="yes")


def test_complete_sample_and_bootstrap_disabled_equivalence(panel):
    d = panel.filter(~((pl.col("id") % 4 == 0) & pl.col("time").is_in([3, 7])))
    complete = d.filter(pl.col("time").count().over("id") == 8)
    kw = dict(
        weights="w",
        anticipation=2,
        multiplier_bootstrap_reps=199,
        multiplier_bootstrap_seed=71,
        **KW,
    )
    a = fit_estimator("y ~ x + z", d, incomplete_units="drop", **kw)
    b = fit_estimator("y ~ x + z", complete, **kw)
    assert_allclose(a.multiplier_draws(), b.multiplier_draws(), atol=1e-11)
    assert_allclose(a.aggregate().multiplier_draws(), b.aggregate().multiplier_draws(), atol=1e-11)
    c = fit_estimator(
        "y ~ 1", panel, multiplier_bootstrap_reps=199, multiplier_bootstrap_seed=71, **KW
    )
    plain = fit_estimator("y ~ 1", panel, **KW)
    assert_allclose(c.beta, plain.beta, atol=1e-11)
    assert_allclose(c.vcov, plain.vcov, atol=1e-11)
    assert_allclose(c.aggregate().vcov, plain.aggregate().vcov, atol=1e-11)


def test_bootstrap_allocation_guard(panel, tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("allocated multiplier state before headroom check")

    monkeypatch.setattr(StreamingMultiplier, "__init__", unexpected)
    with pytest.raises(MemoryError, match="numerical buffers"):
        fit_estimator(
            "y ~ 1",
            panel,
            multiplier_bootstrap_reps=10**8,
            # room for staging, not for 10**8 draws, whatever this process holds
            max_memory_gb=(current_rss_gb() or 0) + 0.6,
            preprocess_mb=1,
            workdir=tmp_path,
            **KW,
        )
    assert not list(tmp_path.iterdir())


def test_pretrend_wald_invariant_to_restriction_units(panel):
    from copy import copy

    f = fit_estimator("y ~ 1", panel, **KW)
    original = f.pretrend_test()
    g = copy(f)
    scaling = np.ones(len(g.beta))
    clean = np.flatnonzero([t < cg for cg, t in g.cells])
    scaling[clean[:2]] = [1e-10, 1e10]
    g.beta = f.beta * scaling
    g.vcov = f.vcov * np.outer(scaling, scaling)
    changed = g.pretrend_test()
    assert_allclose(original["statistic"], changed["statistic"], rtol=1e-10)
    assert_allclose(original["pvalue"], changed["pvalue"], rtol=1e-10)
