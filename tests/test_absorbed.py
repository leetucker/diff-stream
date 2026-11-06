"""Sun-Abraham and BJS with the fixed effects absorbed iteratively.

Against the dense design on small fixed effects, and against R (fixest
sunab, didimputation) on county-by-period effects with 1,200 levels, more
than the dense design takes.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from benchmarks._files import digest, read_csv
from diff_stream import UnimputedObservationsWarning, did_bjs, did_sun_abraham, refit_bootstrap
from diff_stream.simulate import simulate_panel

DATA = Path(__file__).parent / "data"
REF = DATA / "absorbed_reference"
KW = dict(panel=("id", "time"), cohort="first_treat", n_threads=2)
CASES = {
    "county": ("y ~ 1 | county^time", {}),
    "two_terms": ("y ~ x | county^time + industry^time", {}),
    "weighted": ("y ~ x | county^time + industry^time", {"weights": "w"}),
    "unbalanced": ("y ~ x | county^time + industry^time", {"incomplete_units": "keep"}),
    "anticipation": ("y ~ x | county^time + industry^time", {"anticipation": 1}),
    "unimputable": ("y ~ x | county^time + industry^time", {"incomplete_units": "keep"}),
}


@pytest.fixture(scope="module")
def county_panel():
    for name, h in json.loads((REF / "PROVENANCE.json").read_text())["files"].items():
        assert digest(Path(__file__).parents[1] / name) == h, name
    return read_csv(DATA / "absorbed_panel.csv")


def sample(d, name):
    if name == "unbalanced":
        return d.filter(
            ~((pl.col("id") % 7 == 0) & pl.col("time").is_in([3, 6]) & (pl.col("first_treat") > 0))
        )
    if name == "unimputable":
        # never-treated rows missing too: a few county-periods have no untreated rows
        return d.filter(~((pl.col("id") % 7 == 0) & pl.col("time").is_in([3, 6])))
    return d


@pytest.fixture(scope="module")
def small_panel():
    d = simulate_panel(600, 8, seed=5).with_columns((pl.col("id") % 6).alias("region"))
    d = d.with_columns(
        (
            pl.col("id") * 7919 % 101 / 50 * pl.col("time") / 9
            + (pl.col("id") * pl.col("time")) % 3
        ).alias("x"),
        (0.5 + (pl.col("id") % 5) / 4).alias("w"),
    )
    return d.with_columns(
        pl.col("y") + 0.2 * pl.col("x") + 0.3 * pl.col("region") * pl.col("time") / 8
    )


@pytest.mark.parametrize("name", list(CASES))
def test_sun_abraham_matches_fixest(county_panel, name):
    fml, options = CASES[name]
    fit = did_sun_abraham(fml, sample(county_panel, name), **options, **KW)
    assert fit.diagnostics["fixed_effect_solver"] == "iterative"
    expected = read_csv(REF / f"sunab_{name}.csv")
    V = read_csv(REF / f"sunab_{name}_vcov.csv", has_header=False).to_numpy()
    ix = [expected["term"].to_list().index(n) for n in fit.coefnames]
    assert len(ix) == expected.height
    assert_allclose(fit.beta, expected["estimate"].to_numpy()[ix], atol=1e-10, rtol=0)
    assert_allclose(fit.vcov, V[np.ix_(ix, ix)], atol=1e-10, rtol=0)


@pytest.mark.parametrize("name", list(CASES))
def test_bjs_matches_didimputation(county_panel, name):
    fml, options = CASES[name]
    if name == "unimputable":
        # didimputation drops the same observations, silently
        with pytest.warns(UnimputedObservationsWarning, match="6 treated observations"):
            fit = did_bjs(fml, sample(county_panel, name), **options, **KW)
        assert fit.diagnostics["unimputed_observations"] == 6
    else:
        fit = did_bjs(fml, sample(county_panel, name), **options, **KW)
        assert fit.diagnostics["unimputed_observations"] == 0
    assert fit.diagnostics["fixed_effect_solver"] == "iterative"
    expected = read_csv(REF / f"bjs_{name}.csv")
    event = fit.aggregate("event_time")
    labels = [n.split("::")[1] for n in event.coefnames]
    rows = expected.filter(pl.col("term").is_in(labels))
    ix = [rows["term"].to_list().index(n) for n in labels]
    assert len(ix) == len(labels) > 0
    overall = fit.aggregate("overall")
    post = expected.filter(pl.col("term") == "post")
    assert_allclose(event.beta, rows["estimate"].to_numpy()[ix], atol=1e-9, rtol=0)
    assert_allclose(overall.beta, post["estimate"].to_numpy(), atol=1e-9, rtol=0)
    if name != "weighted":  # didimputation's weighted variance is another convention
        assert_allclose(event.se, rows["std.error"].to_numpy()[ix], atol=1e-9, rtol=0)
        assert_allclose(overall.se, post["std.error"].to_numpy(), atol=1e-9, rtol=0)
    fit.cleanup()


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"weights": "w"},
        {"incomplete_units": "keep"},
        {"anticipation": 1},
        {"residual_averaging": "event_time", "leave_out": True},
        {"pretrend_leads": 2},
    ],
    ids=["plain", "weighted", "unbalanced", "anticipation", "leave_out", "pretrend"],
)
@pytest.mark.parametrize("estimator", [did_sun_abraham, did_bjs])
def test_iterative_matches_dense(small_panel, estimator, options):
    if estimator is did_sun_abraham and set(options) & {"residual_averaging", "pretrend_leads"}:
        pytest.skip("BJS options")
    d = small_panel
    if "incomplete_units" in options:
        d = d.filter(~((pl.col("id") % 7 == 0) & pl.col("time").is_in([3, 6])))
    if "anticipation" in options:
        d = d.filter(pl.col("first_treat") != 2)
    fits = [
        estimator("y ~ x | region^time", d, fixed_effect_solver=solver, **options, **KW)
        for solver in ("dense", "iterative")
    ]
    a, b = fits
    assert [f.diagnostics["fixed_effect_solver"] for f in fits] == ["dense", "iterative"]
    assert a.coefnames == b.coefnames
    assert_allclose(b.beta, a.beta, atol=1e-9, rtol=0)
    assert_allclose(b.vcov, a.vcov, atol=1e-10, rtol=0)
    for kind in ("event_time", "overall"):
        x, y = a.aggregate(kind), b.aggregate(kind)
        assert_allclose(y.beta, x.beta, atol=1e-9, rtol=0)
        assert_allclose(y.vcov, x.vcov, atol=1e-10, rtol=0)
    if "pretrend_leads" in options:
        x, y = a.placebos(), b.placebos()
        assert_allclose(y.beta, x.beta, atol=1e-9, rtol=0)
        assert_allclose(y.vcov, x.vcov, atol=1e-10, rtol=0)
        assert a.pretrend_test()["statistic"] == pytest.approx(
            b.pretrend_test()["statistic"], rel=1e-8
        )
    for fit in fits:
        fit.cleanup()


def test_solver_choice(small_panel, county_panel):
    # up to 24 columns (levels and covariates) go to the dense design, above that to
    # the iterative solver; explicit "dense" works to 256 columns
    few = did_sun_abraham("y ~ x | region", small_panel, **KW)  # 6 levels and a covariate
    assert few.diagnostics["fixed_effect_solver"] == "dense"
    edge = did_bjs("y ~ 1 | region^time", small_panel.filter(pl.col("region") < 3), **KW)
    assert edge.diagnostics["fixed_effect_solver"] == "dense"  # 3 regions x 8 periods = 24
    over = did_bjs("y ~ x | region^time", small_panel.filter(pl.col("region") < 3), **KW)
    assert over.diagnostics["fixed_effect_solver"] == "iterative"  # 24 levels and a covariate
    assert (
        did_sun_abraham(
            "y ~ x | region^time", small_panel, fixed_effect_solver="dense", **KW
        ).diagnostics["fixed_effect_solver"]
        == "dense"
    )  # 48 levels, asked for explicitly
    for fit in (few, edge, over):
        fit.cleanup()
    large = did_sun_abraham("y ~ 1 | county^time", county_panel, **KW)
    assert large.diagnostics["fixed_effect_solver"] == "iterative"
    assert large.diagnostics["absorbed_terms"] == ["county^time"]
    with pytest.raises(ValueError, match="256"):
        did_sun_abraham("y ~ 1 | county^time", county_panel, fixed_effect_solver="dense", **KW)
    with pytest.raises(ValueError, match="fixed_effect_solver"):
        did_bjs("y ~ 1 | region^time", small_panel, fixed_effect_solver="sparse", **KW)


@pytest.mark.parametrize("periods", [1, 2])
def test_unimputable_treated_observations(small_panel, periods):
    # region 5 has no untreated observations from period 6 on
    d = small_panel
    if periods == 2:  # evenly spaced labels are reported as given
        d = d.with_columns(
            (1999 + 2 * pl.col("time")).alias("time"),
            pl.when(pl.col("first_treat") == 0)
            .then(0)
            .otherwise(1999 + 2 * pl.col("first_treat"))
            .alias("first_treat"),
        )
    late = (pl.col("region") == 5) & (pl.col("time") >= (6 if periods == 1 else 2011))
    d = d.filter(~(late & (pl.col("first_treat") == 0))).filter(
        ~(late & (pl.col("time") < pl.col("first_treat")))
    )
    options = dict(incomplete_units="keep", **KW)
    fits = []
    for solver in ("dense", "iterative"):
        with pytest.warns(UnimputedObservationsWarning, match="cannot be imputed"):
            fits.append(did_bjs("y ~ 1 | region^time", d, fixed_effect_solver=solver, **options))
    a, b = fits
    assert_allclose(b.beta, a.beta, atol=1e-9, rtol=0)
    assert_allclose(b.vcov, a.vcov, atol=1e-10, rtol=0)
    # the same as dropping those observations by hand
    imputable = d.filter(
        ~(late & (pl.col("first_treat") > 0) & (pl.col("time") >= pl.col("first_treat")))
    )
    c = did_bjs("y ~ 1 | region^time", imputable, fixed_effect_solver="dense", **options)
    assert c.diagnostics["unimputed_observations"] == 0
    assert a.coefnames == c.coefnames
    assert_allclose(a.beta, c.beta, atol=1e-12, rtol=0)
    for kind in ("event_time", "overall"):
        assert_allclose(a.aggregate(kind).vcov, c.aggregate(kind).vcov, atol=1e-12, rtol=0)
    dropped = d.height - imputable.height
    assert a.diagnostics["unimputed_observations"] == dropped
    cells = a.diagnostics["unimputed_cells"]
    assert sum(c["dropped_observations"] for c in cells) == dropped
    times = {c["time"] for c in cells}
    assert min(times) == (6 if periods == 1 else 2011)
    share = a.diagnostics["unimputed_treated_weight_share"]
    assert 0 < share < 1
    for fit in (*fits, c):
        fit.cleanup()


def test_refit_bootstrap(county_panel):
    boot = refit_bootstrap(
        "y ~ 1 | county^time",
        county_panel,
        estimator=did_sun_abraham,
        reps=2,
        seed=1,
        aggregate="overall",
        **KW,
    )
    assert np.isfinite(boot.standard_errors()).all()


def test_batch_rows_follow_the_budget():
    from types import SimpleNamespace

    from diff_stream.absorbed import batch_rows

    def store(headroom):
        return SimpleNamespace(budget=SimpleNamespace(headroom_gb=lambda: headroom))

    assert batch_rows(store(1000), 50) is None  # hdfe-stream's default fits
    rows = batch_rows(store(1), 50)
    assert rows == int(0.25e9 / (32 * 52))
    assert batch_rows(store(0.01), 50) == 50_000
