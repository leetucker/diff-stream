"""trend_sensitivity (Rambachan and Roth's sensitivity analysis for parallel trends)
on diff-stream's event studies, against R's HonestDiD 0.2.8.

tests/data/honestdid_reference/ is written by benchmarks/external/honestdid_reference.R:
HonestDiD's honest_did() on did's Callaway and Sant'Anna event study of the mpdta panel
(universal base period), and on a two-way fixed effects event study of the never-treated
counties and the 2006 cohort fitted by fixest. R's least-favorable critical values are
recorded there and used here, so the comparisons are exact; the folded-normal quantile
is the exact one in both. The engine itself is tested in hdfe-stream
(tests/test_core_sensitivity.py) against every restriction and interval of the package.
"""

import gzip
import json
from pathlib import Path

import hdfe_stream
import numpy as np
import polars as pl
import pytest
from hdfe_stream.core import sensitivity as engine
from numpy.testing import assert_allclose

from benchmarks._files import digest, read_csv
from diff_stream import (
    did_bjs,
    did_callaway_santanna,
    did_dcdh,
    did_extended_twfe,
    did_stacked,
    did_sun_abraham,
    did_two_stage,
)
from diff_stream.simulate import simulate_panel

DATA = Path(__file__).parent / "data"
REF = DATA / "honestdid_reference" / "honestdid.json.gz"
METHOD = {
    "C-LF": "hybrid_least_favorable",
    "Conditional": "conditional",
    "FLCI": "fixed_length",
    "C-F": "hybrid_fixed_length",
}


@pytest.fixture(scope="module")
def ref():
    root = Path(__file__).parents[1]
    for name, h in json.loads((REF.parent / "PROVENANCE.json").read_text())["files"].items():
        assert digest(root / name) == h, name
    with gzip.open(REF, "rt") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def mpdta():
    return read_csv(DATA / "mpdta.csv")


class RecordedLF:
    """R's least-favorable critical values, found by their covariance's fingerprint."""

    def __init__(self, items):
        self.items = items

    def find(self, SA, XA):
        keys = np.array([[it["sum"], it["trace"], it["m"]] for it in self.items], float)
        mine = np.array([SA.sum(), np.trace(SA), len(SA)])
        dist = np.abs(keys - mine).max(axis=1) / (np.abs(mine).max() + 1e-300)
        if XA is not None:
            xs = np.array([np.nan if it["xsum"] is None else it["xsum"] for it in self.items])
            dist = np.where(
                np.isnan(xs), np.inf, dist + np.abs(xs - XA.sum()) / (abs(XA.sum()) + 1)
            )
        j = int(np.argmin(dist))
        assert dist[j] < 1e-6, "no recorded R critical value matches"
        return float(self.items[j]["cv"])


def _patch(monkeypatch, items):
    rec = RecordedLF(items)
    monkeypatch.setattr(engine, "_lf_cv_plain", lambda SA, *a: rec.find(SA, None))
    monkeypatch.setattr(engine, "_lf_cv_arp", lambda SA, XA, *a: rec.find(SA, XA))


def _compare(fit, run, monkeypatch, grid_points, rel=1e-6):
    _patch(monkeypatch, run["lf"])
    r = fit.trend_sensitivity(
        restriction=run["restriction"],
        M=[run["M"]],
        ci_method=METHOD[run["method"]],
        grid_points=grid_points,
    )
    lo, hi = r.table.row(0)[1:]
    width = run["ub"] - run["lb"]
    step = width / 20 if run["method"] != "FLCI" else 0
    if run["method"] == "FLCI":
        # R finds the FLCI's direction with a conic solver, so its center is accurate
        # only to about the square root of the solver's tolerance on the variance
        rel = 2e-4
    # the grid is R's, so equal up to R's and this package's rounding of the covariance
    assert abs(lo - run["lb"]) <= rel * width + 1e-12 or abs(lo - run["lb"]) < step, (lo, run)
    assert abs(hi - run["ub"]) <= rel * width + 1e-12 or abs(hi - run["ub"]) < step, (hi, run)
    return r


def test_callaway_santanna_matches_honest_did(ref, mpdta, monkeypatch):
    fit = did_callaway_santanna(
        "lemp ~ 1",
        mpdta,
        panel=("countyreal", "year"),
        cohort="first.treat",
        base_period="universal",
    )
    es = fit.aggregate()
    want = ref["event_studies"]["cs"]
    assert [int(n.split("::")[1]) for n in es.coefnames] == want["event"]
    assert_allclose(es.beta, want["beta"], rtol=1e-9, atol=1e-12)
    assert_allclose(es.vcov, np.array(want["vcov"]), rtol=1e-7, atol=1e-14)
    runs = [r for r in ref["runs"] if r["source"] == "cs"]
    assert len(runs) == 14
    for run in runs:
        # honest_did tests 100 grid points
        _compare(fit, run, monkeypatch, grid_points=100)


def test_two_way_fixed_effects_event_study_matches_r(ref, mpdta, monkeypatch):
    """No staggered timing: an hdfe-stream regression with leads and lags, the method of
    hdfe-stream's results, against fixest and HonestDiD."""
    d = mpdta.filter(pl.col("first.treat").is_in([0, 2006])).with_columns(
        pl.when(pl.col("first.treat") == 0)
        .then(-1)
        .otherwise(pl.col("year") - pl.col("first.treat"))
        .alias("rel")
    )
    fit = hdfe_stream.feols_stream(
        "lemp ~ i(rel, ref=-1) | countyreal + year",
        d,
        vcov={"CRV1": "countyreal"},
        fe_dof="pyfixest",
        verbose=False,
    )
    want = ref["event_studies"]["twfe"]
    assert [int(n.split("::")[1]) for n in fit.coefnames] == want["event"]
    assert_allclose(fit.beta, want["beta"], rtol=1e-8)
    assert_allclose(fit.vcov, np.array(want["vcov"]), rtol=1e-6)
    runs = [r for r in ref["runs"] if r["source"] == "twfe"]
    for run in runs:
        if "error" in run:
            # R stops ("system is exactly singular") at a grid point; this package
            # computes those points' truncation bounds by the dual route R uses for
            # degenerate solutions
            r = fit.trend_sensitivity(
                restriction=run["restriction"], M=[run["M"]], ci_method=METHOD[run["method"]]
            )
            lo, hi = r.table.row(0)[1:]
            assert np.isfinite(lo) and np.isfinite(hi) and lo < hi
            continue
        _compare(fit, run, monkeypatch, grid_points=1000)


# --------------------------------------------------------------------------
# which fits, and which coefficients
# --------------------------------------------------------------------------

KW = dict(panel=("id", "time"), cohort="first_treat", n_threads=2)


@pytest.fixture(scope="module")
def panel():
    return simulate_panel(600, 8, seed=3)


def _manual(fit_or_es, times_names, reference=-1, **kw):
    names = list(fit_or_es.coefnames)
    ix = [names.index(n) for n in times_names.values()]
    return engine.event_study_sensitivity(
        list(times_names),
        fit_or_es.beta[ix],
        fit_or_es.vcov[np.ix_(ix, ix)],
        reference=reference,
        **kw,
    )


def test_cell_fits_aggregate_first(panel):
    fit = did_callaway_santanna("y ~ 1", panel, base_period="universal", **KW)
    es = fit.aggregate()
    names = {int(n.split("::")[1]): n for n in es.coefnames}
    a = fit.trend_sensitivity(M=[0.5, 1])
    b = es.trend_sensitivity(M=[0.5, 1])
    c = _manual(es, names, M=[0.5, 1])
    assert a.table.equals(b.table) and a.table.equals(c.table)
    assert a.pre == [t for t in sorted(names) if t < -1] and a.post[0] == 0


def test_sun_abraham_extended_twfe_two_stage_and_stacked(panel):
    for fit in (
        did_sun_abraham("y ~ 1", panel, **KW),
        did_extended_twfe("y ~ 1", panel, **KW),
        did_two_stage("y ~ 1", panel, **KW),
        did_stacked("y ~ 1", panel, event_window=(-3, 2), **KW),
    ):
        es = fit.aggregate() if getattr(fit, "cells", None) else fit
        names = {int(n.split("::")[1]): n for n in es.coefnames if "event" in n}
        a = fit.trend_sensitivity(restriction="smoothness", M=[0.01])
        assert a.table.equals(_manual(es, names, restriction="smoothness", M=[0.01]).table)


def test_target_and_horizons(panel):
    fit = did_callaway_santanna("y ~ 1", panel, base_period="universal", **KW)
    r = fit.trend_sensitivity(target="average", horizons=range(-3, 3), M=[1])
    assert r.pre == [-3, -2] and r.post == [0, 1, 2]
    assert_allclose(r.target, [1 / 3] * 3)
    with pytest.raises(ValueError, match="consecutive"):
        fit.trend_sensitivity(horizons=[-4, -2, 0, 1])


def test_dcdh_placebos_are_the_pre_period(panel):
    d = panel.with_columns(
        (pl.col("time") >= pl.col("first_treat")).cast(pl.Int8).alias("d")
        * (pl.col("first_treat") > 0)
    )
    fit = did_dcdh("y ~ 1", d, panel=("id", "time"), treatment="d", effects=3, placebos=2)
    names = {-3: "placebo::2", -2: "placebo::1", 0: "effect::1", 1: "effect::2", 2: "effect::3"}
    a = fit.trend_sensitivity(M=[0.5])
    assert a.table.equals(_manual(fit, names, M=[0.5]).table)
    assert a.pre == [-3, -2] and a.post == [0, 1, 2]
    with pytest.raises(ValueError, match="placebos >= 1"):
        did_dcdh("y ~ 1", d, panel=("id", "time"), treatment="d", effects=2).trend_sensitivity()


def test_refusals(panel):
    with pytest.raises(ValueError, match="base_period='universal'"):
        did_callaway_santanna("y ~ 1", panel, **KW).trend_sensitivity()
    bjs = did_bjs("y ~ 1", panel, pretrend_leads=2, **KW)
    with pytest.raises(ValueError, match="BJS"):
        bjs.trend_sensitivity()
    with pytest.raises(ValueError, match="BJS"):
        bjs.placebos().trend_sensitivity()
    with pytest.raises(ValueError, match="no pre-treatment coefficients"):
        did_extended_twfe("y ~ 1", panel, control_group="not_yet_treated", **KW).trend_sensitivity()
    with pytest.raises(ValueError, match="no event-time coefficients"):
        did_two_stage("y ~ 1", panel, event_study=False, **KW).trend_sensitivity()


def test_plot_event_study(panel, tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    cs = did_callaway_santanna("y ~ 1", panel, base_period="universal", **KW)
    fig = cs.plot_event_study(horizons=range(-3, 3))
    xs = {round(x) for line in fig.axes[0].lines for x in line.get_xdata()}
    assert {-3, -2, -1, 0, 1, 2} <= xs
    fig.savefig(tmp_path / "cs.png")
    d = panel.with_columns(
        (pl.col("time") >= pl.col("first_treat")).cast(pl.Int8).alias("d")
        * (pl.col("first_treat") > 0)
    )
    did_dcdh(
        "y ~ 1", d, panel=("id", "time"), treatment="d", effects=3, placebos=2
    ).plot_event_study()
    did_bjs("y ~ 1", panel, **KW).plot_event_study()  # effects only, no reference period
    did_stacked("y ~ 1", panel, event_window=(-3, 2), **KW).plot_event_study(joint=True, seed=0)
