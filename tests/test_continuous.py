"""Continuous-treatment DiD (did_continuous) against R's contdid and R did.

The references (benchmarks/external/continuous_reference.R) use contdid's only
working settings: not-yet-treated controls and a varying base period (contdid
issue #19). Their curves are compared on a panel whose cells' doses all span the
dose grid, where contdid's evaluation basis is the one it fitted (issue #16). Their
ACRT standard errors use the rescaling of contdid PR #17 (issue #13).
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from benchmarks._files import digest, read_csv
from diff_stream import did_callaway_santanna, did_continuous
from diff_stream.continuous import DoseBasis

DATA = Path(__file__).parent / "data"
REF = DATA / "continuous_reference"
KW = dict(panel=("id", "time"), cohort="first_treat", dose="dose", n_threads=2)
CASES = {
    "cubic": dict(degree=3, interior_knots=0),
    "quadratic_knots": dict(degree=2, interior_knots=2),
}
GRID = np.linspace(1, 10, 46)


@pytest.fixture(scope="module")
def panel():
    for name, h in json.loads((REF / "PROVENANCE.json").read_text())["files"].items():
        assert digest(Path(__file__).parents[1] / name) == h, name
    return read_csv(DATA / "continuous_panel.csv")


@pytest.fixture(scope="module")
def fits(panel):
    return {name: did_continuous("y ~ 1", panel, **KW, **o) for name, o in CASES.items()}


@pytest.mark.parametrize("name", CASES)
def test_overall_matches_contdid(fits, name):
    ref = read_csv(REF / f"overall_{name}.csv")
    fit = fits[name]
    assert fit.coefnames == ["ATT", "ACRT"]
    assert_allclose(fit.beta, ref["estimate"].to_numpy(), rtol=1e-10)
    assert_allclose(fit.se, ref["se"].to_numpy(), rtol=1e-8)
    # The overall ATT is R did's "group" aggregation of the binarized cells.
    did = read_csv(REF / "overall_att_did.csv")
    assert_allclose([fit.beta[0], fit.se[0]], [did["estimate"][0], did["se"][0]], rtol=1e-9)


@pytest.mark.parametrize("name", CASES)
def test_curves_match_contdid(fits, name):
    ref = read_csv(REF / f"curves_{name}.csv")
    assert_allclose(ref["dose"].to_numpy(), GRID)
    level = fits[name].dose_response(dose=GRID)
    slope = fits[name].dose_response("slope", dose=GRID)
    assert_allclose(level["estimate"].to_numpy(), ref["att"].to_numpy(), rtol=1e-9, atol=1e-12)
    assert_allclose(level["std_error"].to_numpy(), ref["att_se"].to_numpy(), rtol=1e-8)
    assert_allclose(slope["estimate"].to_numpy(), ref["acrt"].to_numpy(), rtol=1e-9, atol=1e-12)
    assert_allclose(slope["std_error"].to_numpy(), ref["acrt_se"].to_numpy(), rtol=1e-8)


@pytest.mark.parametrize("name", CASES)
def test_slope_event_study_matches_contdid(fits, name):
    ref = read_csv(REF / f"event_slope_{name}.csv")
    es = fits[name].aggregate(target="slope")
    assert es.coefnames == [f"event::{e}" for e in ref["event"]]
    assert_allclose(es.beta, ref["estimate"].to_numpy(), rtol=1e-9, atol=1e-12)
    assert_allclose(es.se, ref["se"].to_numpy(), rtol=1e-8)


def test_level_event_study_matches_did(fits):
    ref = read_csv(REF / "event_level.csv")
    es = fits["cubic"].aggregate()
    assert_allclose(es.beta, ref["estimate"].to_numpy(), rtol=1e-9, atol=1e-12)
    assert_allclose(es.se, ref["se"].to_numpy(), rtol=1e-8)


def test_binarized_cells_are_callaway_santanna(panel, fits):
    """The level's cells are CS on being treated, whatever the dose spline."""
    cs = did_callaway_santanna(
        "y ~ 1",
        panel,
        panel=("id", "time"),
        cohort="first_treat",
        control_group="not_yet_treated",
        n_threads=2,
    )
    for fit in fits.values():
        att = fit._att
        assert att.coefnames == cs.coefnames
        assert_allclose(att.beta, cs.beta, rtol=1e-12)
        assert_allclose(att.vcov, cs.vcov, rtol=1e-10, atol=1e-16)


def test_linear_spline_is_a_regression_on_the_dose(panel):
    """Degree 1, no knots: each cell is OLS of the outcome change on the dose."""
    fit = did_continuous(
        "y ~ 1", panel, **KW, degree=1, control_group="never_treated", base_period="universal"
    )
    wide = panel.pivot(on="time", index=["id", "first_treat", "dose"], values="y")
    never = wide.filter(pl.col("first_treat") == 0)
    for (g, t), slope in zip(fit._acrt.cells, fit._acrt.beta, strict=True):
        cohort = wide.filter(pl.col("first_treat") == g)
        change = (cohort[str(t)] - cohort[str(g - 1)]).to_numpy()
        X = np.column_stack([np.ones(len(cohort)), cohort["dose"].to_numpy()])
        b, *_ = np.linalg.lstsq(X, change, rcond=None)
        assert_allclose(slope, b[1], rtol=1e-10)
        control = (never[str(t)] - never[str(g - 1)]).to_numpy().mean()
        d = np.array([2.0, 7.5])
        level = (
            fit._basis.values(d)
            @ fit._theta[: len(fit._cells) * 2].reshape(-1, 2)[fit._cells.index((g, t))]
        )
        assert_allclose(level - control, b[0] + b[1] * d - control, rtol=1e-10)


def test_basis_matches_splines2():
    """splines2::bSpline(x, degree = 2, knots = c(3, 6), Boundary.knots = c(1, 10)) and dbs."""
    basis = DoseBasis([3.0, 6.0], 1.0, 10.0, 2)
    x = np.array([1.0, 2.5, 3.0, 5.0, 9.99, 10.0])
    values = basis.values(x)
    assert_allclose(values[:, 0], 1.0)
    assert_allclose(values[:, 1:].sum(axis=1) + _first(x), 1.0, rtol=1e-12)
    assert_allclose(basis.slopes(x)[:, 0], 0.0)
    with pytest.raises(ValueError, match="within the treated doses"):
        basis.values([0.5])


def _first(x):
    # the first B-spline (dropped by intercept = FALSE): (1 - (x - 1) / 2)^2 on [1, 3]
    return np.where(x < 3, (1 - (x - 1) / 2) ** 2, 0.0)


def test_uniform_band_is_wider(fits):
    fit = did_continuous(
        "y ~ 1",
        read_csv(DATA / "continuous_panel.csv"),
        **KW,
        multiplier_bootstrap_reps=500,
        multiplier_bootstrap_seed=3,
    )
    pointwise = fit.dose_response("slope")
    uniform = fit.dose_response("slope", uniform=True)
    assert_allclose(uniform["estimate"], pointwise["estimate"])
    assert np.all(
        (uniform["ci_high"] - uniform["ci_low"]).to_numpy()
        > (pointwise["ci_high"] - pointwise["ci_low"]).to_numpy()
    )
    with pytest.raises(ValueError, match="multiplier_bootstrap_reps"):
        fits["cubic"].dose_response(uniform=True)


def test_results_do_not_depend_on_threads_or_batches(panel, fits):
    one = did_continuous("y ~ 1", panel, **{**KW, "n_threads": 1}, batch_units=300)
    assert_allclose(one.beta, fits["cubic"].beta, rtol=1e-12)
    assert_allclose(one.vcov, fits["cubic"].vcov, rtol=1e-10)
    assert_allclose(
        one.dose_response()["std_error"], fits["cubic"].dose_response()["std_error"], rtol=1e-10
    )


def test_default_grid_and_knots_follow_contdid(panel, fits):
    treated = panel.filter((pl.col("time") == 1) & (pl.col("first_treat") > 0))["dose"].to_numpy()
    grid = fits["cubic"].dose_response()["dose"].to_numpy()
    assert_allclose(grid, np.quantile(treated, np.arange(10, 100) / 100))
    assert_allclose(
        fits["quadratic_knots"].diagnostics["knots"], np.quantile(treated, [1 / 3, 2 / 3])
    )


def test_errors(panel):
    with pytest.raises(ValueError, match="no covariates"):
        did_continuous("y ~ x", panel, **KW)
    zero = panel.with_columns(
        pl.when(pl.col("id") == panel.filter(pl.col("first_treat") > 0)["id"][0])
        .then(0.0)
        .otherwise(pl.col("dose"))
        .alias("dose")
    )
    with pytest.raises(ValueError, match="positive dose"):
        did_continuous("y ~ 1", zero, **KW)
    varying = panel.with_columns(
        pl.when((pl.col("first_treat") > 0) & (pl.col("time") == 6))
        .then(pl.col("dose") + 1)
        .otherwise(pl.col("dose"))
        .alias("dose")
    )
    with pytest.raises(ValueError, match="constant within each unit"):
        did_continuous("y ~ 1", varying, **KW)
    with pytest.raises(ValueError, match="repeated knots"):
        did_continuous(
            "y ~ 1", panel.with_columns(pl.col("dose").round(0)), **KW, interior_knots=30
        )
    with pytest.raises(ValueError, match="incomplete_units"):
        did_continuous("y ~ 1", panel, **KW, incomplete_units="keep")


def test_treatment_column_and_spaced_periods(panel, fits):
    treated = (pl.col("first_treat") > 0) & (pl.col("time") >= pl.col("first_treat"))
    by_treatment = did_continuous(
        "y ~ 1",
        panel.with_columns(treated.cast(pl.Int8).alias("d")),
        panel=("id", "time"),
        treatment="d",
        dose="dose",
        n_threads=2,
    )
    assert_allclose(by_treatment.beta, fits["cubic"].beta, rtol=1e-12)
    assert_allclose(by_treatment.vcov, fits["cubic"].vcov, rtol=1e-10)
    spaced = did_continuous(
        "y ~ 1", panel.with_columns(pl.col("time") * 2, pl.col("first_treat") * 2), **KW
    )
    assert_allclose(spaced.beta, fits["cubic"].beta, rtol=1e-12)
    assert spaced.cohort_time()["Coefficient"][0] == "g::6:t::4"
    assert spaced.diagnostics["time_max"] == 12


def test_profile(panel):
    p = did_continuous("y ~ 1", panel, **KW, profile=True)
    assert p.model == "diff-stream did_continuous"
    assert p.notes["spline_columns"] == 4


@pytest.mark.resource
def test_constrained_fit_matches_unconstrained_fit(resource_runner, resource_enforcement, tmp_path):
    """Under a memory watchdog, with small staging pieces and batches, the same fit."""
    import os
    import sys

    options = dict(preprocess_mb=4, batch_units=37, n_threads=2)
    code = f"""
import json
from benchmarks._files import read_csv
from diff_stream import did_continuous
r = did_continuous("y ~ 1", read_csv({str(DATA / "continuous_panel.csv")!r}), panel=("id", "time"),
    cohort="first_treat", dose="dose", interior_knots=2, workdir={str(tmp_path / "child")!r},
    **{options!r})
print(json.dumps(dict(beta=r.beta.tolist(), vcov=r.vcov.tolist(),
    curve=r.dose_response()["std_error"].to_list(), event=r.aggregate(target="slope").se.tolist())))
"""
    threads = dict.fromkeys(
        [
            "STREAM_LIMITS_MAX_THREADS",
            "POLARS_MAX_THREADS",
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "NUMBA_NUM_THREADS",
        ],
        "2",
    )
    env = dict(os.environ, **threads, PYTHONPATH=str(Path(__file__).parents[1]))
    result = resource_runner([sys.executable, "-c", code], memory_gb=1.5, env=env, timeout=120)
    assert result.status == "ok", result.as_dict()
    assert result.mode == resource_enforcement["mode"]
    fitted = json.loads(result.stdout)
    baseline = did_continuous(
        "y ~ 1", read_csv(DATA / "continuous_panel.csv"), **KW, interior_knots=2
    )
    assert_allclose(fitted["beta"], baseline.beta, atol=1e-10, rtol=0)
    assert_allclose(fitted["vcov"], baseline.vcov, atol=1e-12, rtol=0)
    assert_allclose(fitted["curve"], baseline.dose_response()["std_error"], rtol=1e-9)
    assert_allclose(fitted["event"], baseline.aggregate(target="slope").se, rtol=1e-9)


# A discrete dose: diff-diff's saturated regression (R's contdid has none) and a
# dense oracle.

DISCRETE = DATA / "continuous_discrete_reference"
DISCRETE_KW = dict(KW, dose_type="discrete")
CONTROLS = {"never_treated": "never_treated", "not_yet_treated": "not_yet_treated"}


@pytest.fixture(scope="module")
def discrete_panel():
    for name, h in json.loads((DISCRETE / "PROVENANCE.json").read_text())["files"].items():
        assert digest(Path(__file__).parents[1] / name) == h, name
    return read_csv(DATA / "continuous_discrete_panel.csv")


@pytest.fixture(scope="module")
def discrete_fits(discrete_panel):
    return {
        cg: did_continuous("y ~ 1", discrete_panel, **DISCRETE_KW, control_group=cg)
        for cg in CONTROLS
    }


class Dual:
    """A value and its gradient with respect to each unit's weight, at weights of 1."""

    def __init__(self, value, grad):
        self.v, self.g = value, grad

    def __add__(self, other):
        o = _dual(other, self.g)
        return Dual(self.v + o.v, self.g + o.g)

    __radd__ = __add__

    def __sub__(self, other):
        o = _dual(other, self.g)
        return Dual(self.v - o.v, self.g - o.g)

    def __mul__(self, other):
        o = _dual(other, self.g)
        return Dual(self.v * o.v, self.g * o.v + o.g * self.v)

    __rmul__ = __mul__

    def __truediv__(self, other):
        o = _dual(other, self.g)
        return Dual(self.v / o.v, (self.g - self.v / o.v * o.g) / o.v)


def _dual(x, like):
    return x if isinstance(x, Dual) else Dual(x, np.zeros_like(like))


def _mean(x, mask):
    grad = np.zeros(len(x))
    grad[mask] = (x[mask] - x[mask].mean()) / mask.sum()
    return Dual(x[mask].mean(), grad)


def _count(mask):
    return Dual(mask.sum(), mask.astype(float))


def discrete_oracle(panel, control_group, *, fixed_shares=False):
    """The discrete-dose estimates from dense means, with gradient-based standard errors.

    Every estimate is a smooth function of unit-weighted means, so its influence
    function is its gradient with respect to the unit weights, and its standard
    error the root sum of squares of that gradient. The overall ACRT and the
    curves average the cells with fixed weights (contdid's), the overall ATT with
    the estimated cohort shares (R did's). ``fixed_shares=True`` also holds the
    cohort shares and each cohort's dose shares fixed.
    """
    wide = panel.pivot(on="time", index=["id", "first_treat", "dose"], values="y").sort("id")
    times = sorted(int(c) for c in wide.columns[3:])
    Y = wide.select([str(t) for t in times]).to_numpy()
    G, D = wide["first_treat"].to_numpy(), wide["dose"].to_numpy()
    cohorts, levels = np.unique(G[G > 0]), np.unique(D[G > 0])
    gap = np.diff(np.r_[0.0, levels])
    cells = [(g, t) for g in cohorts for t in times[1:]]
    att, acrt, curve_att, curve_acrt = [], [], [], []
    for g, t in cells:
        base = g - 1 if t >= g else t - 1
        change = Y[:, times.index(t)] - Y[:, times.index(base)]
        control = (G == 0) | ((G > t) & (G != g)) if control_group == "not_yet_treated" else G == 0
        mean_control = _mean(change, control)
        mine = G == g
        at = [_mean(change, mine & (D == d)) - mean_control for d in levels]
        slopes = [(at[j] - (at[j - 1] if j else 0.0)) / gap[j] for j in range(len(levels))]
        shares = [_count(mine & (D == d)) / _count(mine) for d in levels]
        if fixed_shares:
            shares = [s.v for s in shares]
        att.append(_mean(change, mine) - mean_control)
        acrt.append(sum(s * a for s, a in zip(shares, slopes, strict=True)))
        curve_att.append(at)
        curve_acrt.append(slopes)
    sizes = {g: _count(G == g) for g in cohorts}
    total = sum(sizes.values())
    weights = []
    for g, t in cells:
        n_post = sum(c == g and s >= c for c, s in cells)
        weights.append(sizes[g] / total * (1.0 / n_post) if t >= g else 0.0)
    fixed = [w.v if isinstance(w, Dual) else w for w in weights]
    overall_weights = fixed if fixed_shares else weights

    def average(values, w):
        return sum(wi * v for wi, v in zip(w, values, strict=True) if not np.all(wi == 0))

    out = [average(att, overall_weights), average(acrt, fixed)]
    out += [average([c[j] for c in curve_att], fixed) for j in range(len(levels))]
    out += [average([c[j] for c in curve_acrt], fixed) for j in range(len(levels))]
    estimate = np.array([o.v for o in out])
    se = np.array([np.sqrt((o.g**2).sum()) for o in out])
    return levels, estimate, se


@pytest.mark.parametrize("cg", CONTROLS)
def test_discrete_dose_matches_the_dense_oracle(discrete_panel, discrete_fits, cg):
    fit = discrete_fits[cg]
    levels, estimate, se = discrete_oracle(discrete_panel, cg)
    J = len(levels)
    level, slope = fit.dose_response(), fit.dose_response("slope")
    assert_allclose(level["dose"].to_numpy(), levels)
    assert_allclose(fit.beta, estimate[:2], rtol=1e-12)
    assert_allclose(fit.se, se[:2], rtol=1e-10)
    assert_allclose(level["estimate"].to_numpy(), estimate[2 : 2 + J], rtol=1e-12)
    assert_allclose(level["std_error"].to_numpy(), se[2 : 2 + J], rtol=1e-10)
    assert_allclose(slope["estimate"].to_numpy(), estimate[2 + J :], rtol=1e-12)
    assert_allclose(slope["std_error"].to_numpy(), se[2 + J :], rtol=1e-10)


@pytest.mark.parametrize("cg", CONTROLS)
def test_discrete_dose_matches_diff_diff(discrete_panel, discrete_fits, cg):
    fit = discrete_fits[cg]
    overall = read_csv(DISCRETE / f"overall_{cg}.csv")
    assert_allclose(fit.beta, overall["estimate"].to_numpy(), rtol=1e-12)
    # diff-diff holds the cohort shares and each cohort's dose shares fixed in the
    # overall standard errors; with those fixed, the oracle gives its numbers.
    _, _, fixed_se = discrete_oracle(discrete_panel, cg, fixed_shares=True)
    assert_allclose(fixed_se[:2], overall["se"].to_numpy(), rtol=1e-10)
    curves = read_csv(DISCRETE / f"curves_{cg}.csv")
    level, slope = fit.dose_response(), fit.dose_response("slope")
    assert_allclose(level["dose"].to_numpy(), curves["dose"].to_numpy())
    for frame, name in ((level, "att"), (slope, "acrt")):
        assert_allclose(frame["estimate"].to_numpy(), curves[name].to_numpy(), rtol=1e-12)
        assert_allclose(frame["std_error"].to_numpy(), curves[f"{name}_se"].to_numpy(), rtol=1e-10)
    cells = read_csv(DISCRETE / f"cells_{cg}.csv")
    assert fit._att.cells == list(zip(cells["g"], cells["t"], strict=True))
    assert_allclose(fit._att.beta, cells["att"].to_numpy(), rtol=1e-10, atol=1e-13)
    assert_allclose(fit._acrt.beta, cells["acrt"].to_numpy(), rtol=1e-10, atol=1e-13)
    M, K = len(fit._cells), fit._basis.width
    beta, control = fit._theta[: M * K].reshape(M, K), fit._theta[M * K :]
    doses = level["dose"].to_numpy()
    values, slopes = fit._basis.values(doses), fit._basis.slopes(doses)
    for j in range(M):
        att_d = values @ beta[j] - control[j]
        acrt_d = slopes @ beta[j] - fit._basis.slope_control(doses) * control[j]
        reference = cells[j]
        assert_allclose(att_d, [reference[f"att_{d:g}"][0] for d in doses], atol=1e-12)
        assert_allclose(acrt_d, [reference[f"acrt_{d:g}"][0] for d in doses], atol=1e-12)
    event = read_csv(DISCRETE / f"event_{cg}.csv")
    es = fit.aggregate()
    assert es.coefnames == [f"event::{e}" for e in event["event"]]
    assert_allclose(es.beta, event["estimate"].to_numpy(), rtol=1e-10, atol=1e-13)
    # diff-diff's event-study standard errors also hold the cohort shares fixed
    # (ours add R did's estimated-share term, as test_level_event_study_matches_did)
    sizes = {int(g): n for g, n in fit.diagnostics["cohort_sizes"].items()}
    fixed = []
    for e in event["event"]:
        w = np.array([sizes[g] * (t - g == e) for g, t in fit._att.cells], dtype=float)
        w /= w.sum()
        fixed.append(np.sqrt(w @ fit._att.vcov @ w))
    assert_allclose(fixed, event["se"].to_numpy(), rtol=1e-10)


def test_a_single_dose_level_is_the_binary_effect(discrete_panel):
    """With every treated unit at dose 1, ACRT = ATT(1) = the binarized ATT."""
    binary = discrete_panel.with_columns(
        pl.when(pl.col("first_treat") > 0).then(1.0).otherwise(0.0).alias("dose")
    )
    fit = did_continuous("y ~ 1", binary, **DISCRETE_KW)
    assert_allclose(fit.beta[1], fit.beta[0], rtol=1e-12)
    assert_allclose(fit._acrt.beta, fit._att.beta, rtol=1e-12, atol=1e-14)
    assert_allclose(fit._acrt.vcov, fit._att.vcov, rtol=1e-10, atol=1e-16)
    assert_allclose(fit.dose_response()["estimate"], [fit.beta[0]], rtol=1e-12)
    cs = did_callaway_santanna(
        "y ~ 1",
        binary,
        panel=("id", "time"),
        cohort="first_treat",
        control_group="not_yet_treated",
        n_threads=2,
    )
    assert_allclose(fit._att.beta, cs.beta, rtol=1e-12)


def test_discrete_dose_does_not_depend_on_threads_or_batches(discrete_panel, discrete_fits):
    base = discrete_fits["not_yet_treated"]
    one = did_continuous(
        "y ~ 1", discrete_panel, **{**DISCRETE_KW, "n_threads": 1}, batch_units=250
    )
    assert_allclose(one.beta, base.beta, rtol=1e-12)
    assert_allclose(one.vcov, base.vcov, rtol=1e-10)
    assert_allclose(
        one.dose_response("slope")["std_error"],
        base.dose_response("slope")["std_error"],
        rtol=1e-10,
    )
    banded = did_continuous(
        "y ~ 1",
        discrete_panel,
        **DISCRETE_KW,
        multiplier_bootstrap_reps=300,
        multiplier_bootstrap_seed=1,
    )
    pointwise, uniform = banded.dose_response("slope"), banded.dose_response("slope", uniform=True)
    assert np.all(
        (uniform["ci_high"] - uniform["ci_low"]).to_numpy()
        > (pointwise["ci_high"] - pointwise["ci_low"]).to_numpy()
    )


def test_discrete_dose_errors(panel, discrete_panel, discrete_fits):
    with pytest.raises(ValueError, match="no spline"):
        did_continuous("y ~ 1", discrete_panel, **DISCRETE_KW, degree=2)
    with pytest.raises(ValueError, match="no spline"):
        did_continuous("y ~ 1", discrete_panel, **DISCRETE_KW, interior_knots=1)
    with pytest.raises(ValueError, match="dose_type"):
        did_continuous("y ~ 1", discrete_panel, **{**KW, "dose_type": "binary"})
    with pytest.raises(ValueError, match="only at its dose levels"):
        discrete_fits["never_treated"].dose_response(dose=[3.0])
    with pytest.raises(ValueError, match="only at its dose levels"):
        did_continuous("y ~ 1", discrete_panel, **DISCRETE_KW, dose_grid=[1.5])
    missing = discrete_panel.filter(~((pl.col("first_treat") == 4) & (pl.col("dose") == 7)))
    with pytest.raises(ValueError, match="cohort 4 has no units at dose 7"):
        did_continuous("y ~ 1", missing, **DISCRETE_KW)
    with pytest.raises(ValueError, match="at most 1000 values"):
        did_continuous("y ~ 1", panel, **DISCRETE_KW)
    fit = discrete_fits["never_treated"]
    assert fit.diagnostics["dose_levels"] == [1.0, 2.0, 4.0, 7.0]
    assert "discrete dose: 4 levels; control group: never_treated" in fit._summary_lines()
    p = did_continuous("y ~ 1", discrete_panel, **DISCRETE_KW, profile=True)
    assert p.notes["dose_levels"] == 4
