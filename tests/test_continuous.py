"""Continuous-treatment DiD (did_continuous) against R's contdid and R did.

The references (benchmarks/external/continuous_reference.R) use contdid's only
working settings: not-yet-treated controls and a varying base period (contdid
issue #19). Their curves are compared on a panel whose cells' doses all span the
dose grid, where contdid's evaluation basis is the one it fitted (issue #16). Their
ACRT standard errors use the rescaling of contdid PR #17 (issue #13).
"""

import json
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from benchmarks._files import digest, read_csv
from diff_stream import ExperimentalWarning, did_callaway_santanna, did_continuous
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
    # the formula is checked before the fit warns that covariates are experimental
    with pytest.raises(ValueError, match="unknown variables"):
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
        did_continuous("y ~ 1", panel, **KW, incomplete_units="partly")


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
import warnings
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


# Covariates and an absorbed fixed effect (experimental): diff-diff's regression
# adjustment, Callaway-Sant'Anna's, and a dense calculation of the standard errors.

COVARIATES = DATA / "continuous_covariates_reference"
COVARIATE_FITS = {
    "continuous": ("y ~ x | region", dict(dose="dose")),
    "discrete": ("y_level ~ x | region", dict(dose="dose_level", dose_type="discrete")),
}
COVARIATE_KW = dict(panel=("id", "time"), cohort="first_treat", n_threads=2)
COVARIATE_GRID = np.linspace(1, 10, 19)
SMALL = 600


def fit_quietly(*args, **options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        return did_continuous(*args, **options)


@pytest.fixture(scope="module")
def covariate_panel():
    for name, h in json.loads((COVARIATES / "PROVENANCE.json").read_text())["files"].items():
        assert digest(Path(__file__).parents[1] / name) == h, name
    return read_csv(DATA / "continuous_covariates_panel.csv")


@pytest.fixture(scope="module")
def covariate_fits(covariate_panel):
    return {
        (kind, cg): fit_quietly(fml, covariate_panel, **COVARIATE_KW, **o, control_group=cg)
        for kind, (fml, o) in COVARIATE_FITS.items()
        for cg in CONTROLS
    }


def covariate_oracle(panel, fit, outcome, dose, control_group, *, fixed_shares=False):
    """The adjusted overall ATT and ACRT from dense regressions, with standard errors
    from their complex-step gradient in the unit weights.

    Each cell regresses the comparison units' change on an intercept, x at the base
    period and region dummies, and fits the treated units' adjusted changes on the
    fit's dose basis. ``fixed_shares=True`` holds the cohort shares and each cohort's
    average slope of the basis fixed, as diff-diff does.
    """

    def wide(v):
        return panel.pivot(on="time", index="id", values=v).sort("id").drop("id").to_numpy()

    Y, X = wide(outcome), wide("x")
    units = panel.filter(pl.col("time") == 1).sort("id")
    G, D, R = (units[c].to_numpy() for c in ("first_treat", dose, "region"))
    dummies = (R[:, None] == np.unique(R)[None, 1:]).astype(float)
    cohorts, basis = np.unique(G[G > 0]), fit._basis
    cells = [(g, t) for g in cohorts for t in range(2, Y.shape[1] + 1)]
    Psi, dPsi = np.zeros((len(G), basis.width)), np.zeros((len(G), basis.width))
    Psi[G > 0], dPsi[G > 0] = basis.values(D[G > 0]), basis.slopes(D[G > 0])
    slopes = [dPsi[G == g].mean(axis=0) for g, t in cells] if fixed_shares else None

    def estimate(u, fixed=None):
        att, acrt = [], []
        for j, (g, t) in enumerate(cells):
            b = g - 1 if t >= g else t - 1
            change = Y[:, t - 1] - Y[:, b - 1]
            if control_group == "not_yet_treated":
                control = (G == 0) | ((G > t) & (G != g))
            else:
                control = G == 0
            Z = np.column_stack([np.ones(len(G)), X[:, min(t, b) - 1], dummies])
            Zc, uc = Z[control], u[control]
            gamma = np.linalg.solve(Zc.T @ (uc[:, None] * Zc), Zc.T @ (uc * change[control]))
            mine = G == g
            a, um, P = change[mine] - Z[mine] @ gamma, u[mine], Psi[mine]
            beta = np.linalg.solve(P.T @ (um[:, None] * P), P.T @ (um * a))
            att.append((um * a).sum() / um.sum())
            if slopes is None:
                acrt.append((um * (dPsi[mine] @ beta)).sum() / um.sum())
            else:
                acrt.append(slopes[j] @ beta)
        n = {g: u[G == g].sum() for g in cohorts}
        total = sum(n.values())
        w = np.zeros(len(cells), dtype=u.dtype)
        for g in cohorts:
            post = np.array([c == g and t >= g for c, t in cells])
            w[post] = n[g] / total / post.sum()
        cell_weights = w.real if fixed is None else fixed
        return np.r_[(cell_weights if fixed_shares else w) @ att, cell_weights @ acrt], w.real

    base, w = estimate(np.ones(len(G)))
    grads = []
    for i in range(len(G)):
        u = np.ones(len(G), dtype=complex)
        u[i] += 1e-30j
        grads.append(estimate(u, fixed=w)[0].imag / 1e-30)
    return base, np.sqrt((np.array(grads) ** 2).sum(axis=0))


@pytest.mark.parametrize("cg", CONTROLS)
@pytest.mark.parametrize("kind", COVARIATE_FITS)
def test_covariates_match_diff_diff(covariate_fits, kind, cg):
    fit, name = covariate_fits[kind, cg], f"{kind}_{cg}"
    overall = read_csv(COVARIATES / f"overall_{name}.csv")
    assert_allclose(fit.beta, overall["estimate"].to_numpy(), rtol=1e-12)
    curves = read_csv(COVARIATES / f"curves_{name}.csv")
    dose = COVARIATE_GRID if kind == "continuous" else None
    level, slope = fit.dose_response(dose=dose), fit.dose_response("slope", dose=dose)
    assert_allclose(level["dose"].to_numpy(), curves["dose"].to_numpy())
    for frame, target in ((level, "att"), (slope, "acrt")):
        assert_allclose(frame["estimate"].to_numpy(), curves[target].to_numpy(), rtol=1e-11)
        assert_allclose(
            frame["std_error"].to_numpy(), curves[f"{target}_se"].to_numpy(), rtol=1e-10
        )
    cells = read_csv(COVARIATES / f"cells_{name}.csv")
    assert fit._att.cells == list(zip(cells["g"], cells["t"], strict=True))
    assert_allclose(fit._att.beta, cells["att"].to_numpy(), rtol=1e-10, atol=1e-13)
    assert_allclose(fit._acrt.beta, cells["acrt"].to_numpy(), rtol=1e-10, atol=1e-13)
    event = read_csv(COVARIATES / f"event_{name}.csv")
    assert_allclose(fit.aggregate().beta, event["estimate"].to_numpy(), rtol=1e-10, atol=1e-13)


@pytest.mark.parametrize("cg", CONTROLS)
@pytest.mark.parametrize("kind", COVARIATE_FITS)
def test_covariates_match_the_dense_oracle(covariate_panel, kind, cg):
    """Every overall standard error, on the first 600 units; with the shares fixed,
    diff-diff's on the same units."""
    fml, options = COVARIATE_FITS[kind]
    small = covariate_panel.filter(pl.col("id") < SMALL)
    fit = fit_quietly(fml, small, **COVARIATE_KW, **options, control_group=cg)
    outcome = fml.split("~")[0].strip()
    estimate, se = covariate_oracle(small, fit, outcome, options["dose"], cg)
    assert_allclose(fit.beta, estimate, rtol=1e-12)
    assert_allclose(fit.se, se, rtol=1e-10)
    _, fixed = covariate_oracle(small, fit, outcome, options["dose"], cg, fixed_shares=True)
    reference = read_csv(COVARIATES / f"overall_small_{kind}_{cg}.csv")
    assert_allclose(fit.beta, reference["estimate"].to_numpy(), rtol=1e-12)
    assert_allclose(fixed, reference["se"].to_numpy(), rtol=1e-10)


@pytest.mark.parametrize("kind", COVARIATE_FITS)
def test_an_absorbed_fixed_effect_equals_its_dummies(covariate_panel, covariate_fits, kind):
    fml, options = COVARIATE_FITS[kind]
    dense = fit_quietly(
        fml.replace("| region", "+ C(region)"), covariate_panel, **COVARIATE_KW, **options
    )
    absorbed = covariate_fits[kind, "not_yet_treated"]
    assert_allclose(absorbed.beta, dense.beta, rtol=1e-12)
    assert_allclose(absorbed.vcov, dense.vcov, rtol=1e-10)
    assert_allclose(
        absorbed.dose_response("slope")["std_error"],
        dense.dose_response("slope")["std_error"],
        rtol=1e-10,
    )


@pytest.mark.parametrize(
    "fml,base_period",
    [("y ~ x | region", "varying"), ("y ~ x | region", "universal"), ("y ~ 1 | region", "varying")],
)
@pytest.mark.parametrize("cg", CONTROLS)
def test_binarized_covariate_cells_are_callaway_santanna(covariate_panel, fml, base_period, cg):
    """The effect of being treated is Callaway-Sant'Anna's regression adjustment,
    including base-period covariates and a fixed effect constant over time."""
    fit = fit_quietly(
        fml, covariate_panel, **COVARIATE_KW, dose="dose", control_group=cg, base_period=base_period
    )
    cs = did_callaway_santanna(
        fml.replace("| region", "+ C(region)").replace("1 + ", ""),
        covariate_panel,
        panel=("id", "time"),
        cohort="first_treat",
        control_group=cg,
        base_period=base_period,
        adjustment="regression",
        n_threads=2,
    )
    assert fit._att.coefnames == cs.coefnames
    assert_allclose(fit._att.beta, cs.beta, rtol=1e-10, atol=1e-13)
    assert_allclose(fit._att.vcov, cs.vcov, rtol=1e-9, atol=1e-16)


def test_treated_units_without_comparisons_in_their_level_are_left_out(covariate_panel):
    """With never-treated comparisons, leaving units out equals dropping them (the
    default dose grid aside, which comes from every treated unit's dose)."""
    grid = np.linspace(2, 9, 8)
    lonely = covariate_panel.filter(pl.col("first_treat") == 3)["id"].unique().sort()[:5]
    marked = covariate_panel.with_columns(
        pl.when(pl.col("id").is_in(lonely.implode()))
        .then(99)
        .otherwise(pl.col("region"))
        .alias("region")
    )
    options = dict(**COVARIATE_KW, dose="dose", control_group="never_treated")
    with pytest.warns(ExperimentalWarning):
        fit = did_continuous("y ~ x | region", marked, **options)
    assert fit.diagnostics["treated_units_left_out"] == {3: 5, 4: 0, 5: 0}
    assert any("5 treated units are left out" in w for w in fit.diagnostics["warnings"])
    dropped = fit_quietly(
        "y ~ x | region", marked.filter(~pl.col("id").is_in(lonely.implode())), **options
    )
    assert_allclose(fit.beta, dropped.beta, rtol=1e-12)
    assert_allclose(fit._att.vcov, dropped._att.vcov, rtol=1e-10)
    assert_allclose(
        fit.dose_response(dose=grid)["std_error"],
        dropped.dose_response(dose=grid)["std_error"],
        rtol=1e-10,
    )


def test_covariate_fits_do_not_depend_on_threads_or_batches(covariate_panel, covariate_fits):
    base = covariate_fits["continuous", "not_yet_treated"]
    one = fit_quietly(
        "y ~ x | region",
        covariate_panel,
        **{**COVARIATE_KW, "n_threads": 1},
        dose="dose",
        batch_units=300,
    )
    assert_allclose(one.beta, base.beta, rtol=1e-12)
    assert_allclose(one.vcov, base.vcov, rtol=1e-10)
    banded = fit_quietly(
        "y ~ x | region",
        covariate_panel,
        **COVARIATE_KW,
        dose="dose",
        multiplier_bootstrap_reps=300,
        multiplier_bootstrap_seed=2,
    )
    pointwise, uniform = banded.dose_response("slope"), banded.dose_response("slope", uniform=True)
    assert np.all(
        (uniform["ci_high"] - uniform["ci_low"]).to_numpy()
        > (pointwise["ci_high"] - pointwise["ci_low"]).to_numpy()
    )


def test_covariates_warn_and_are_reported(covariate_panel):
    import hdfe_stream

    assert issubclass(ExperimentalWarning, hdfe_stream.ExperimentalWarning)
    with pytest.warns(
        ExperimentalWarning, match="covariates and a fixed effect goes beyond"
    ) as caught:
        fit = did_continuous("y ~ x | region", covariate_panel, **COVARIATE_KW, dose="dose")
    assert caught[0].filename == __file__
    assert "continuous.html#experimental-extensions" in fit.diagnostics["warnings"][0]
    with pytest.warns(ExperimentalWarning, match="with covariates goes beyond"):
        did_continuous("y ~ x", covariate_panel, **COVARIATE_KW, dose="dose")
    assert fit.diagnostics["covariate_names"] == ["x"]
    assert fit.diagnostics["fixed_effect"] == "region"
    assert fit.diagnostics["fixed_effect_levels"] == [8]
    assert (
        fit._summary_lines()[-1]
        == "covariates: x; fixed effect: region (experimental: see the docs)"
    )


@pytest.mark.parametrize(
    "fml,message",
    [
        ("y ~ x | id", "already differenced out"),
        ("y ~ x | time", "already differs by period"),
        ("y ~ x | region + region", "given twice"),
        ("y ~ x | region + id", "already differenced out"),
        ("y ~ x | region + C(industry)", "cannot read the fixed effect"),
        ("y ~ x | region | x", "at most one"),
        ("y ~ dose + x", "not as a covariate"),
        ("y ~ x | nowhere", "unknown fixed-effect columns"),
        ("y ~ C(region) | region", "collinear with the fixed effect"),
    ],
)
def test_covariate_errors(covariate_panel, fml, message):
    with pytest.raises(ValueError, match=message):
        fit_quietly(fml, covariate_panel, **COVARIATE_KW, dose="dose")


def test_fixed_effect_with_missing_levels_is_an_error(covariate_panel):
    gaps = covariate_panel.with_columns(
        pl.when(pl.col("id") == 7).then(None).otherwise(pl.col("region")).alias("region")
    )
    with pytest.raises(ValueError, match="missing values"):
        fit_quietly("y ~ x | region", gaps, **COVARIATE_KW, dose="dose")


# The lowest-dose comparison (every unit treated, at one date), against diff-diff's
# control_group="lowest_dose", and the general fit against a dense calculation.

LOWEST = DATA / "continuous_lowest_reference"
LOWEST_GRID = np.linspace(1, 9.5, 18)
LOWEST_KW = dict(
    panel=("id", "time"), cohort="first_treat", control_group="lowest_dose", n_threads=2
)
LOWEST_FITS = {
    "continuous_varying": ("y ~ 1", dict(dose="dose"), LOWEST_GRID),
    "continuous_universal": ("y ~ 1", dict(dose="dose", base_period="universal"), LOWEST_GRID),
    "discrete_varying": ("y_level ~ 1", dict(dose="dose_level", dose_type="discrete"), None),
    "discrete_universal": (
        "y_level ~ 1",
        dict(dose="dose_level", dose_type="discrete", base_period="universal"),
        None,
    ),
    "reference_varying": (
        "y ~ 1",
        dict(dose="dose", reference_dose=3.0),
        LOWEST_GRID[LOWEST_GRID > 3],
    ),
}


@pytest.fixture(scope="module")
def lowest_panel():
    for name, h in json.loads((LOWEST / "PROVENANCE.json").read_text())["files"].items():
        assert digest(Path(__file__).parents[1] / name) == h, name
    return read_csv(DATA / "continuous_lowest_panel.csv")


@pytest.fixture(scope="module")
def lowest_fits(lowest_panel):
    return {
        name: did_continuous(fml, lowest_panel, **LOWEST_KW, **o)
        for name, (fml, o, _) in LOWEST_FITS.items()
    }


def general_oracle(
    panel,
    fit,
    outcome,
    dose,
    control_group,
    *,
    reference=None,
    covariates=False,
    fixed_effect=False,
    weights=None,
    cluster=None,
    fixed_slopes=False,
    fixed_shares=False,
    gradients=False,
):
    """The overall ATT and ACRT from dense regressions, with standard errors from their
    complex-step gradient in each unit's weight (summed by ``cluster`` if given).

    Each cell's comparison units observed in its two periods have their change
    regressed, with their weights in the period compared, on an intercept, x at the
    covariate period (``covariates``) and region dummies (``fixed_effect=True``, or
    dummies of each of the unit-level columns it lists); the treated units observed
    in it have their adjusted changes fitted on the fit's dose basis. ``reference(d)``
    marks the lowest-dose comparison's reference units. Treated units whose design
    is not estimable from the comparison units (with one term: whose region has
    none) in a cell they are observed in are left out of their cohort. ``fixed_slopes`` holds each cell's average slope
    of the basis fixed, as diff-diff does for the overall ACRT, and ``fixed_shares``
    the cohort shares in the overall ATT, as it does there. ``gradients=True``
    returns the units' (or clusters') gradients in place of the standard errors.
    """
    ids = panel["id"].unique().sort()

    def wide(v):
        frame = panel.pivot(on="time", index="id", values=v).sort("id")
        times = sorted(int(c) for c in frame.columns if c != "id")
        return frame.select([str(t) for t in times]).to_numpy().astype(float)

    Y = wide(outcome)
    X = wide("x") if covariates else None
    Omega = wide(weights) if weights is not None else np.where(np.isfinite(Y), 1.0, np.nan)
    units = panel.group_by("id").first().sort("id")
    assert units["id"].to_list() == ids.to_list()
    G, D = (units[c].to_numpy() for c in ("first_treat", dose))
    terms = ["region"] if fixed_effect is True else list(fixed_effect or [])
    levels = [units[c].to_numpy() for c in terms]
    if reference is not None:
        G = np.where(reference(D), 0, G)
    basis, universal = fit._basis, fit.diagnostics["base_period"] == "universal"
    cells = fit._cells
    treated = G > 0
    Psi, dPsi = np.zeros((len(G), basis.width)), np.zeros((len(G), basis.width))
    Psi[treated], dPsi[treated] = basis.values(D[treated]), basis.slopes(D[treated])
    anchor = np.nanmean(Omega, axis=1)

    def frame(g, t):
        b = g - 1 if (t >= g or universal) else t - 1
        observed = np.isfinite(Y[:, t - 1]) & np.isfinite(Y[:, b - 1])
        if control_group == "not_yet_treated":
            control = (G == 0) | ((G > max(t, b)) & (G != g))
        else:
            control = G == 0
        return b, observed, control & observed

    def design(g, t):
        """The cell's design (its comparison units' independent columns) and which of
        the cohort's units observed in the cell have an estimable design."""
        b, observed, control = frame(g, t)
        columns = [np.ones(len(G))]
        if covariates:
            columns.append(X[:, min(t, b) - 1])
        for R in levels:
            columns += [(R == r).astype(float) for r in np.unique(R[control])]
        Z = np.column_stack(columns)
        Zc = Z[control]
        independent = []
        for c in range(Z.shape[1]):
            if np.linalg.matrix_rank(Zc[:, independent + [c]]) > len(independent):
                independent.append(c)
        _, s, vt = np.linalg.svd(Zc, full_matrices=True)
        null = vt[np.sum(s > 1e-9 * s[0]) :].T
        mine = (G == g) & observed
        estimable = np.zeros(len(G), dtype=bool)
        estimable[mine] = np.abs(Z[mine] @ null).max(axis=1, initial=0) < 1e-8
        # a level without comparison units has no column: such units are not estimable
        for R in levels:
            estimable[mine] &= np.isin(R[mine], R[control])
        return Z[:, independent], mine, estimable

    designs = {(g, t): design(g, t) for g, t in cells}
    kept = np.ones(len(G), dtype=bool)
    for mine_design in designs.values():
        _, mine, estimable = mine_design
        kept[mine & ~estimable] = False

    def estimate(u, fixed=None):
        att, acrt, slopes = [], [], []
        for g, t in cells:
            b, observed, control = frame(g, t)
            om = Omega[:, t - 1] * u
            change = Y[:, t - 1] - Y[:, b - 1]
            Z = designs[(g, t)][0]
            Zc, wc = Z[control], om[control]
            gamma = np.linalg.solve(Zc.T @ (wc[:, None] * Zc), Zc.T @ (wc * change[control]))
            mine = (G == g) & observed & kept
            a, wm, P = change[mine] - Z[mine] @ gamma, om[mine], Psi[mine]
            beta = np.linalg.solve(P.T @ (wm[:, None] * P), P.T @ (wm * a))
            att.append((wm * a).sum() / wm.sum())
            slope = (wm[:, None] * dPsi[mine]).sum(axis=0) / wm.sum()
            slopes.append(slope.real)
            held = fixed is not None and fixed[1] is not None
            acrt.append((fixed[1][len(slopes) - 1] if held else slope) @ beta)
        cohorts = sorted({g for g, t in cells})
        n = {g: (anchor * u)[(G == g) & kept].sum() for g in cohorts}
        total = sum(n.values())
        w = np.zeros(len(cells), dtype=u.dtype)
        for g in cohorts:
            post = np.array([c == g and t >= g for c, t in cells])
            w[post] = n[g] / total / post.sum()
        cell_weights = w.real if fixed is None else fixed[0]
        att_weights = cell_weights if fixed_shares else w
        return np.r_[att_weights @ np.array(att), cell_weights @ np.array(acrt)], (w.real, slopes)

    base, fixed = estimate(np.ones(len(G)))
    grads = []
    for i in range(len(G)):
        u = np.ones(len(G), dtype=complex)
        u[i] += 1e-30j
        grads.append(estimate(u, fixed=fixed if fixed_slopes else (fixed[0], None))[0].imag)
    grads = np.array(grads) / 1e-30
    if cluster is not None:
        labels = units[cluster].to_numpy()
        grads = np.array([grads[labels == c].sum(axis=0) for c in np.unique(labels)])
    if gradients:
        return base, grads
    return base, np.sqrt((grads**2).sum(axis=0))


def _without_base_rows(frame):
    """diff-diff lists the universal base period as a cell (and event) with effect 0."""
    return frame.filter((pl.col("att") != 0) | (pl.col("acrt") != 0))


@pytest.mark.parametrize("name", LOWEST_FITS)
def test_lowest_dose_matches_diff_diff(lowest_panel, lowest_fits, name):
    fit, (fml, options, grid) = lowest_fits[name], LOWEST_FITS[name]
    overall = read_csv(LOWEST / f"overall_{name}.csv")
    assert_allclose(fit.beta, overall["estimate"].to_numpy(), rtol=1e-12)
    assert_allclose(fit.se[0], overall["se"][0], rtol=1e-10)
    # diff-diff holds each cell's average slope of the basis fixed in the overall
    # ACRT's standard error; with it fixed, the dense calculation gives its number
    reference = (lambda d: d <= 3.0) if "reference_dose" in options else (lambda d: d == d.min())
    outcome = fml.split("~")[0].strip()
    _, fixed = general_oracle(
        lowest_panel,
        fit,
        outcome,
        options["dose"],
        "lowest_dose",
        reference=reference,
        fixed_slopes=True,
    )
    assert_allclose(fixed[1], overall["se"][1], rtol=1e-10)
    curves = read_csv(LOWEST / f"curves_{name}.csv")
    level, slope = fit.dose_response(dose=grid), fit.dose_response("slope", dose=grid)
    assert_allclose(level["dose"].to_numpy(), curves["dose"].to_numpy())
    for frame, target in ((level, "att"), (slope, "acrt")):
        assert_allclose(frame["estimate"].to_numpy(), curves[target].to_numpy(), rtol=1e-11)
        assert_allclose(
            frame["std_error"].to_numpy(), curves[f"{target}_se"].to_numpy(), rtol=1e-10
        )
    cells = _without_base_rows(read_csv(LOWEST / f"cells_{name}.csv"))
    assert fit._att.cells == list(zip(cells["g"], cells["t"], strict=True))
    assert_allclose(fit._att.beta, cells["att"].to_numpy(), rtol=1e-10, atol=1e-13)
    assert_allclose(fit._acrt.beta, cells["acrt"].to_numpy(), rtol=1e-10, atol=1e-13)
    event = read_csv(LOWEST / f"event_{name}.csv").filter(pl.col("se") != 0)
    es = fit.aggregate()
    assert es.coefnames == [f"event::{e}" for e in event["event"]]
    assert_allclose(es.beta, event["estimate"].to_numpy(), rtol=1e-10, atol=1e-13)
    assert_allclose(es.se, event["se"].to_numpy(), rtol=1e-10)


@pytest.mark.parametrize("name", ["continuous_varying", "discrete_universal", "reference_varying"])
def test_lowest_dose_matches_the_dense_oracle(lowest_panel, lowest_fits, name):
    fit, (fml, options, _) = lowest_fits[name], LOWEST_FITS[name]
    reference = (lambda d: d <= 3.0) if "reference_dose" in options else (lambda d: d == d.min())
    estimate, se = general_oracle(
        lowest_panel,
        fit,
        fml.split("~")[0].strip(),
        options["dose"],
        "lowest_dose",
        reference=reference,
    )
    assert_allclose(fit.beta, estimate, rtol=1e-12)
    assert_allclose(fit.se, se, rtol=1e-10)


def test_lowest_dose_reference_is_reported(lowest_fits):
    lowest, ranged = lowest_fits["continuous_varying"], lowest_fits["reference_varying"]
    assert lowest.diagnostics["control_group"] == "lowest_dose"
    assert lowest.diagnostics["reference_dose"] == 0.5
    assert lowest.diagnostics["reference_units"] == 19
    assert lowest.diagnostics["cohort_sizes"] == {"0": 19, "4": 711}
    assert lowest.diagnostics["warnings"] == []
    assert "control group: units at the lowest dose, 0.5" in lowest._summary_lines()[-1]
    assert ranged.diagnostics["reference"] == "doses up to"
    assert "units with doses up to 3" in ranged._summary_lines()[-1]
    # the reference units are never on the curve
    with pytest.raises(ValueError, match="within the treated doses' range"):
        ranged.dose_response(dose=[2.0])
    assert lowest_fits["discrete_varying"].diagnostics["dose_levels"] == [2.0, 4.0, 7.0]


@pytest.mark.parametrize(
    "fml,covariates",
    [("y ~ 1 | region", False), ("y ~ x | region", True)],
)
def test_lowest_dose_with_a_fixed_effect_matches_the_dense_oracle(lowest_panel, fml, covariates):
    """Region-by-period intercepts estimated from each region's reference units."""
    with pytest.warns(ExperimentalWarning, match="fixed effect goes beyond"):
        fit = did_continuous(fml, lowest_panel, **LOWEST_KW, dose="dose", reference_dose=3.0)
    assert sum(fit.diagnostics["treated_units_left_out"].values()) == 0
    estimate, se = general_oracle(
        lowest_panel,
        fit,
        "y",
        "dose",
        "lowest_dose",
        reference=lambda d: d <= 3.0,
        covariates=covariates,
        fixed_effect=True,
    )
    assert_allclose(fit.beta, estimate, rtol=1e-12)
    assert_allclose(fit.se, se, rtol=1e-10)
    dummies = fit_quietly(
        fml.replace("| region", "+ C(region)"),
        lowest_panel,
        **LOWEST_KW,
        dose="dose",
        reference_dose=3.0,
    )
    assert_allclose(dummies.beta, fit.beta, rtol=1e-11)
    assert_allclose(dummies.se, fit.se, rtol=1e-10)
    for target in ("level", "slope"):
        a, b = fit.dose_response(target), dummies.dose_response(target)
        assert_allclose(a["estimate"].to_numpy(), b["estimate"].to_numpy(), rtol=1e-10, atol=1e-12)
        assert_allclose(a["std_error"].to_numpy(), b["std_error"].to_numpy(), rtol=1e-9)


def test_regions_without_reference_units_are_left_out(lowest_panel):
    """At the lowest dose only some institutions have reference units."""
    fit = fit_quietly("y ~ 1 | region", lowest_panel, **LOWEST_KW, dose="dose")
    left = fit.diagnostics["treated_units_left_out"][4]
    reference_regions = lowest_panel.filter((pl.col("time") == 1) & (pl.col("dose") == 0.5))[
        "region"
    ].unique()
    expected = lowest_panel.filter(
        (pl.col("time") == 1)
        & (pl.col("dose") > 0.5)
        & ~pl.col("region").is_in(reference_regions.implode())
    ).height
    assert left == expected > 0
    assert "left out" in fit.diagnostics["warnings"][-1]
    estimate, se = general_oracle(
        lowest_panel,
        fit,
        "y",
        "dose",
        "lowest_dose",
        reference=lambda d: d == 0.5,
        fixed_effect=True,
    )
    assert_allclose(fit.beta, estimate, rtol=1e-12)
    assert_allclose(fit.se, se, rtol=1e-10)


def general_fit(fml, data, **options):
    """did_continuous through its general fit, even where the unadjusted one applies."""
    from diff_stream import _options
    from diff_stream.continuous import _continuous

    return _continuous(fml, data, force_general=True, **_options.internal(options))


@pytest.mark.parametrize("cg", CONTROLS)
@pytest.mark.parametrize("kind", ["continuous", "discrete"])
def test_the_general_fit_reproduces_the_unadjusted_fit(panel, discrete_panel, kind, cg):
    data, options = (panel, KW) if kind == "continuous" else (discrete_panel, DISCRETE_KW)
    options = dict(
        options, control_group=cg, multiplier_bootstrap_reps=50, multiplier_bootstrap_seed=3
    )
    plain, general = did_continuous("y ~ 1", data, **options), general_fit("y ~ 1", data, **options)
    assert_allclose(general.beta, plain.beta, rtol=1e-12)
    assert_allclose(general.vcov, plain.vcov, rtol=1e-10)
    for target in ("level", "slope"):
        a = plain.dose_response(target, uniform=True)
        b = general.dose_response(target, uniform=True)
        for column in ("estimate", "std_error", "ci_low", "ci_high"):
            assert_allclose(b[column].to_numpy(), a[column].to_numpy(), rtol=1e-10, atol=1e-12)
        p_cells, g_cells = plain._target(target), general._target(target)
        assert_allclose(g_cells.beta, p_cells.beta, rtol=1e-10, atol=1e-13)
        assert_allclose(g_cells.se, p_cells.se, rtol=1e-10)
        for kind_ in ("event_time", "overall", "cohort", "calendar"):
            p, g = plain.aggregate(kind_, target=target), general.aggregate(kind_, target=target)
            assert_allclose(g.beta, p.beta, rtol=1e-10, atol=1e-13)
            assert_allclose(g.se, p.se, rtol=1e-10)


@pytest.mark.parametrize(
    "change,options,message",
    [
        (None, dict(cohort="first_treat", control_group="never_treated"), None),
        ("never", dict(), "without untreated units"),
        ("cohorts", dict(), "single treatment date"),
        ("single", dict(), "at least 2 units at the reference"),
        (None, dict(reference_dose=0.4), "no units have a dose above|at least 2"),
        (None, dict(reference_dose=3.0, dose_type="discrete"), "lowest level"),
        (None, dict(reference_dose=-1.0), "positive dose"),
    ],
)
def test_lowest_dose_errors(lowest_panel, change, options, message):
    data = lowest_panel
    if change == "never":
        data = data.with_columns(
            pl.when(pl.col("id") < 5).then(0).otherwise(pl.col("first_treat")).alias("first_treat")
        )
    elif change == "cohorts":
        data = data.with_columns(
            pl.when(pl.col("id") < 100)
            .then(5)
            .otherwise(pl.col("first_treat"))
            .alias("first_treat")
        )
    elif change == "single":
        first = data.filter(pl.col("dose") == 0.5)["id"].unique().sort()
        data = data.filter(~pl.col("id").is_in(first[1:].implode()))
    kw = dict(LOWEST_KW, dose="dose") | options
    if message is None:
        # never-treated controls are another comparison: no reference_dose there
        with pytest.raises(ValueError, match="at least one never-treated"):
            did_continuous("y ~ 1", data, **kw)
        with pytest.raises(ValueError, match="only to control_group='lowest_dose'"):
            did_continuous("y ~ 1", data, **(kw | dict(reference_dose=3.0)))
        return
    with pytest.raises(ValueError, match=message):
        did_continuous("y ~ 1", data, **kw)


def test_lowest_dose_does_not_depend_on_threads_or_batches(lowest_panel):
    options = dict(
        LOWEST_KW,
        dose="dose",
        reference_dose=3.0,
        multiplier_bootstrap_reps=40,
        multiplier_bootstrap_seed=1,
    )
    one = fit_quietly("y ~ x | region", lowest_panel, **(options | dict(n_threads=1)))
    many = fit_quietly(
        "y ~ x | region", lowest_panel, **(options | dict(n_threads=4, batch_units=97))
    )
    assert_allclose(many.beta, one.beta, rtol=1e-12)
    assert_allclose(many.vcov, one.vcov, rtol=1e-10)
    a, b = one.dose_response(uniform=True), many.dose_response(uniform=True)
    assert_allclose(b["ci_low"].to_numpy(), a["ci_low"].to_numpy(), rtol=1e-10)


# Unbalanced panels (experimental): each comparison uses the units observed in both
# of its periods. No package implements this, so the dense calculation is the check.


def _unbalanced(panel, seed):
    """The panel with 8% of unit-periods missing, and the first two periods of 5% of units."""
    rng = np.random.default_rng(seed)
    late = rng.random(panel["id"].max() + 1) < 0.05
    keep = (rng.random(panel.height) > 0.08) & ~(
        late[panel["id"].to_numpy()] & (panel["time"].to_numpy() <= 2)
    )
    return panel.filter(pl.Series(keep))


@pytest.mark.parametrize(
    "fml,covariates,fixed_effect",
    [("y ~ 1", False, False), ("y ~ x | region", True, True)],
)
@pytest.mark.parametrize("cg", CONTROLS)
def test_unbalanced_panel_matches_the_dense_oracle(
    covariate_panel, fml, covariates, fixed_effect, cg
):
    small = _unbalanced(covariate_panel.filter(pl.col("id") < SMALL), 5)
    with pytest.warns(ExperimentalWarning, match="an unbalanced panel goes beyond"):
        fit = did_continuous(
            fml, small, **COVARIATE_KW, dose="dose", control_group=cg, incomplete_units="keep"
        )
    assert fit.diagnostics["balanced"] is False
    estimate, se = general_oracle(
        small, fit, "y", "dose", cg, covariates=covariates, fixed_effect=fixed_effect
    )
    assert_allclose(fit.beta, estimate, rtol=1e-12)
    assert_allclose(fit.se, se, rtol=1e-10)


def test_unbalanced_lowest_dose_with_a_fixed_effect_matches_the_dense_oracle(lowest_panel):
    data = _unbalanced(lowest_panel, 6)
    with pytest.warns(ExperimentalWarning, match="a fixed effect and an unbalanced panel"):
        fit = did_continuous(
            "y_level ~ 1 | region",
            data,
            **LOWEST_KW,
            dose="dose_level",
            dose_type="discrete",
            incomplete_units="keep",
        )
    estimate, se = general_oracle(
        data,
        fit,
        "y_level",
        "dose_level",
        "lowest_dose",
        reference=lambda d: d == 1.0,
        fixed_effect=True,
    )
    assert_allclose(fit.beta, estimate, rtol=1e-12)
    assert_allclose(fit.se, se, rtol=1e-10)
    # each cell's treated units are those observed in both its periods
    cells = fit._att
    assert cells.observed_counts.max() <= fit.diagnostics["cohort_sizes"]["4"]
    assert cells.observed_counts.min() < cells.observed_counts.max()


def test_a_balanced_panel_kept_is_the_balanced_fit(covariate_panel, covariate_fits):
    """incomplete_units="keep" on a balanced panel is neither different nor experimental."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExperimentalWarning)
        kept = did_continuous(
            "y ~ 1", covariate_panel, **COVARIATE_KW, dose="dose", incomplete_units="keep"
        )
    plain = did_continuous("y ~ 1", covariate_panel, **COVARIATE_KW, dose="dose")
    assert kept.diagnostics["balanced"] is True
    assert_allclose(kept.beta, plain.beta, rtol=1e-14)
    assert_allclose(kept.vcov, plain.vcov, rtol=1e-14)


def test_unbalanced_fit_does_not_depend_on_threads_or_batches(covariate_panel):
    data = _unbalanced(covariate_panel, 7)
    options = dict(COVARIATE_KW, dose="dose", incomplete_units="keep")
    one = fit_quietly("y ~ x | region", data, **(options | dict(n_threads=1)))
    many = fit_quietly("y ~ x | region", data, **(options | dict(n_threads=4, batch_units=101)))
    assert_allclose(many.beta, one.beta, rtol=1e-12)
    assert_allclose(many.vcov, one.vcov, rtol=1e-10)


# Sampling weights, against diff-diff's survey weights. diff-diff's weighted standard
# errors apply each unit's weight twice (upstream-issues, issue 6): the estimates are
# compared directly, the standard errors with the dense calculation, and diff-diff's
# are reproduced from the same per-unit scores with the weight applied again.

WEIGHTED = DATA / "continuous_weights_reference"
WEIGHTED_FITS = {
    "lowest_continuous": ("lowest", "y ~ 1", dict(dose="dose", control_group="lowest_dose")),
    "lowest_discrete": (
        "lowest",
        "y_level ~ 1",
        dict(dose="dose_level", dose_type="discrete", control_group="lowest_dose"),
    ),
    "staggered_never_treated": (
        "staggered",
        "y ~ 1",
        dict(dose="dose", control_group="never_treated"),
    ),
    "staggered_not_yet_treated": (
        "staggered",
        "y ~ 1",
        dict(dose="dose", control_group="not_yet_treated"),
    ),
}


@pytest.fixture(scope="module")
def weighted_panels(lowest_panel, covariate_panel):
    for name, h in json.loads((WEIGHTED / "PROVENANCE.json").read_text())["files"].items():
        assert digest(Path(__file__).parents[1] / name) == h, name
    return {
        "lowest": lowest_panel,
        "staggered": covariate_panel.with_columns(weight=1 + (pl.col("id") % 7) / 2),
    }


def _twice(weights, grads):
    """diff-diff's standard error: the scores times the (mean-one) weight again, centered."""
    s = (weights / weights.mean())[:, None] * grads
    n = len(weights)
    return np.sqrt(((s - s.mean(axis=0)) ** 2).sum(axis=0) * n / (n - 1))


@pytest.mark.parametrize("name", WEIGHTED_FITS)
def test_weights_match_diff_diff(weighted_panels, name):
    which, fml, options = WEIGHTED_FITS[name]
    data = weighted_panels[which]
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExperimentalWarning)  # constant weights are not experimental
        fit = did_continuous(
            fml, data, panel=("id", "time"), cohort="first_treat", weights="weight", **options
        )
    assert fit.diagnostics["weights_column"] == "weight"
    overall = read_csv(WEIGHTED / f"overall_{name}.csv")
    assert_allclose(fit.beta, overall["estimate"].to_numpy(), rtol=1e-12)
    grid = LOWEST_GRID if options.get("dose_type") != "discrete" else None
    curves = read_csv(WEIGHTED / f"curves_{name}.csv")
    for target, column in (("level", "att"), ("slope", "acrt")):
        frame = fit.dose_response(target, dose=grid)
        assert_allclose(frame["estimate"].to_numpy(), curves[column].to_numpy(), rtol=1e-11)
    cells = read_csv(WEIGHTED / f"cells_{name}.csv")
    assert fit._att.cells == list(zip(cells["g"], cells["t"], strict=True))
    assert_allclose(fit._att.beta, cells["att"].to_numpy(), rtol=1e-10, atol=1e-13)
    assert_allclose(fit._acrt.beta, cells["acrt"].to_numpy(), rtol=1e-10, atol=1e-13)
    event = read_csv(WEIGHTED / f"event_{name}.csv")
    assert_allclose(fit.aggregate().beta, event["estimate"].to_numpy(), rtol=1e-10, atol=1e-13)
    outcome, dose = fml.split("~")[0].strip(), options["dose"]
    cg = options["control_group"]
    reference = (lambda d: d == d.min()) if cg == "lowest_dose" else None
    oracle = dict(reference=reference, weights="weight")
    estimate, se = general_oracle(data, fit, outcome, dose, cg, **oracle)
    assert_allclose(fit.beta, estimate, rtol=1e-12)
    assert_allclose(fit.se, se, rtol=1e-10)
    # diff-diff's: the weight twice, the average slope held fixed (and, with several
    # cohorts, the cohort shares: only its ACRT is reproduced there)
    _, grads = general_oracle(
        data, fit, outcome, dose, cg, **oracle, fixed_slopes=True, gradients=True
    )
    weights = data.group_by("id").first().sort("id")["weight"].to_numpy()
    twice = _twice(weights, grads)
    reproduced = slice(0, 2) if which == "lowest" else slice(1, 2)
    assert_allclose(twice[reproduced], overall["se"].to_numpy()[reproduced], rtol=1e-10)


@pytest.mark.parametrize("cg", CONTROLS)
def test_weighted_binarized_cells_are_callaway_santanna(weighted_panels, cg):
    """The effect of being treated equals weighted Callaway-Sant'Anna (validated against
    R did), standard errors included."""
    data = weighted_panels["staggered"]
    options = dict(panel=("id", "time"), cohort="first_treat", control_group=cg, weights="weight")
    fit = did_continuous("y ~ 1", data, dose="dose", **options)
    cs = did_callaway_santanna("y ~ 1", data, **options)
    assert fit._att.cells == list(cs.cells)
    assert_allclose(fit._att.beta, cs.beta, rtol=1e-12, atol=1e-14)
    assert_allclose(fit._att.se, cs.se, rtol=1e-10)
    assert_allclose(fit.aggregate().se, cs.aggregate().se, rtol=1e-10)


def test_integer_weights_are_replicated_units(lowest_panel):
    """A weight of 2 counts a unit twice in every estimate."""
    weights = (1 + pl.col("id") % 3).cast(pl.Float64)
    data = lowest_panel.with_columns(w=weights)
    copies = pl.concat(
        [
            data.filter(pl.col("w") >= k + 1).with_columns(pl.col("id") + k * 100_000)
            for k in range(3)
        ]
    )
    options = dict(LOWEST_KW, dose="dose", reference_dose=3.0)
    weighted = fit_quietly("y ~ x | region", data, **options, weights="w")
    replicated = fit_quietly("y ~ x | region", copies, **options)
    assert_allclose(weighted.beta, replicated.beta, rtol=1e-12)
    grid = LOWEST_GRID[LOWEST_GRID > 3]  # the default grid is unweighted quantiles
    for target in ("level", "slope"):
        assert_allclose(
            weighted.dose_response(target, dose=grid)["estimate"].to_numpy(),
            replicated.dose_response(target, dose=grid)["estimate"].to_numpy(),
            rtol=1e-10,
        )


@pytest.mark.parametrize("unbalanced", [False, True])
def test_time_varying_weights_match_the_dense_oracle(lowest_panel, unbalanced):
    """Each cohort-period weights its units by their weights in the period compared."""
    data = _unbalanced(lowest_panel, 8) if unbalanced else lowest_panel
    with pytest.warns(ExperimentalWarning, match="time-varying weights"):
        fit = did_continuous(
            "y ~ 1 | region",
            data,
            **LOWEST_KW,
            dose="dose",
            reference_dose=3.0,
            weights="weight_t",
            allow_time_varying_weights=True,
            incomplete_units="keep",
        )
    assert fit.diagnostics["weights_time_invariant"] is False
    estimate, se = general_oracle(
        data,
        fit,
        "y",
        "dose",
        "lowest_dose",
        reference=lambda d: d <= 3.0,
        fixed_effect=True,
        weights="weight_t",
    )
    assert_allclose(fit.beta, estimate, rtol=1e-12)
    assert_allclose(fit.se, se, rtol=1e-10)


def test_time_varying_weights_need_permission(lowest_panel):
    with pytest.raises(ValueError, match="constant within unit"):
        did_continuous("y ~ 1", lowest_panel, **LOWEST_KW, dose="dose", weights="weight_t")


# Standard errors clustered above the unit, against diff-diff's survey PSUs. Its overall
# ACRT holds the average slope fixed, and with several cohorts its overall ATT and event
# study hold the cohort shares fixed; those are reproduced with the dense calculation.

CLUSTERED = DATA / "continuous_cluster_reference"
CLUSTERED_FITS = {
    "reference_continuous": (
        "lowest",
        "y ~ 1",
        dict(dose="dose", control_group="lowest_dose", reference_dose=3.0, cluster="major"),
    ),
    "lowest_discrete": (
        "lowest",
        "y_level ~ 1",
        dict(dose="dose_level", dose_type="discrete", control_group="lowest_dose", cluster="major"),
    ),
    "staggered_never_treated": (
        "staggered",
        "y ~ 1",
        dict(dose="dose", control_group="never_treated", cluster="region"),
    ),
    "staggered_not_yet_treated": (
        "staggered",
        "y ~ 1",
        dict(dose="dose", control_group="not_yet_treated", cluster="region"),
    ),
}


@pytest.fixture(scope="module")
def clustered_panels(lowest_panel, covariate_panel):
    for name, h in json.loads((CLUSTERED / "PROVENANCE.json").read_text())["files"].items():
        assert digest(Path(__file__).parents[1] / name) == h, name
    return {"lowest": lowest_panel, "staggered": covariate_panel}


@pytest.mark.parametrize("name", CLUSTERED_FITS)
def test_clusters_match_diff_diff(clustered_panels, name):
    which, fml, options = CLUSTERED_FITS[name]
    data = clustered_panels[which]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # clustering is not experimental; no one-cluster cell
        fit = did_continuous(
            fml, data, panel=("id", "time"), cohort="first_treat", **options, vcov="CRV1"
        )
    G = data[options["cluster"]].n_unique()
    assert fit.n_clusters == {options["cluster"]: G} and fit.df_t == G - 1
    assert fit.diagnostics["inference"].startswith(f"clustered by {options['cluster']}")
    overall = read_csv(CLUSTERED / f"overall_{name}.csv")
    assert_allclose(fit.beta, overall["estimate"].to_numpy(), rtol=1e-12)
    grid = LOWEST_GRID if options.get("dose_type") != "discrete" else None
    if "reference_dose" in options:
        grid = grid[grid > 3]
    curves = read_csv(CLUSTERED / f"curves_{name}.csv")
    for target, column in (("level", "att"), ("slope", "acrt")):
        frame = fit.dose_response(target, dose=grid)
        assert_allclose(frame["estimate"].to_numpy(), curves[column].to_numpy(), rtol=1e-11)
        assert_allclose(
            frame["std_error"].to_numpy(), curves[f"{column}_se"].to_numpy(), rtol=1e-10
        )
    cells = read_csv(CLUSTERED / f"cells_{name}.csv")
    assert_allclose(fit._att.beta, cells["att"].to_numpy(), rtol=1e-10, atol=1e-13)
    event = read_csv(CLUSTERED / f"event_{name}.csv")
    es = fit.aggregate()
    assert_allclose(es.beta, event["estimate"].to_numpy(), rtol=1e-10, atol=1e-13)
    outcome, cg = fml.split("~")[0].strip(), options["control_group"]
    if cg == "lowest_dose":
        cut = options.get("reference_dose")
        reference = (lambda d: d <= cut) if cut else (lambda d: d == d.min())
    else:
        reference = None
    oracle = dict(reference=reference, cluster=options["cluster"])
    # ours: the dense calculation, clustered (CRV0 there, so the factor here)
    estimate, se = general_oracle(data, fit, outcome, options["dose"], cg, **oracle)
    assert_allclose(fit.beta, estimate, rtol=1e-12)
    assert_allclose(fit.se, se * np.sqrt(G / (G - 1)), rtol=1e-10)
    # diff-diff's: the average slope (and the cohort shares) held fixed
    _, fixed = general_oracle(
        data, fit, outcome, options["dose"], cg, **oracle, fixed_slopes=True, fixed_shares=True
    )
    assert_allclose(fixed * np.sqrt(G / (G - 1)), overall["se"].to_numpy(), rtol=1e-10)
    sizes = {int(g): n for g, n in fit.diagnostics["cohort_sizes"].items()}
    held = []
    for e in event["event"]:
        w = np.array([sizes[g] * (t - g == e) for g, t in fit._att.cells], dtype=float)
        w /= w.sum()
        held.append(np.sqrt(w @ fit._att.vcov @ w))
    assert_allclose(held, event["se"].to_numpy(), rtol=1e-10)


def test_clustered_weighted_fit_matches_the_dense_oracle(lowest_panel):
    """Clusters, weights, a fixed effect, covariates and missing periods together."""
    data = _unbalanced(lowest_panel, 9)
    fit = fit_quietly(
        "y ~ x | region",
        data,
        **LOWEST_KW,
        dose="dose",
        reference_dose=3.0,
        weights="weight",
        cluster="major",
        incomplete_units="keep",
        multiplier_bootstrap_reps=30,
    )
    estimate, se = general_oracle(
        data,
        fit,
        "y",
        "dose",
        "lowest_dose",
        reference=lambda d: d <= 3.0,
        covariates=True,
        fixed_effect=True,
        weights="weight",
        cluster="major",
    )
    assert_allclose(fit.beta, estimate, rtol=1e-12)
    assert_allclose(fit.se, se, rtol=1e-10)
    # CRV1 uses the number of clusters, also after with_vcov and in aggregations
    crv1 = fit.with_vcov("CRV1")
    assert_allclose(crv1.se, se * np.sqrt(40 / 39), rtol=1e-10)
    assert_allclose(crv1.aggregate().se, fit.aggregate().se * np.sqrt(40 / 39), rtol=1e-12)
    band = fit.dose_response(uniform=True)
    assert (band["ci_high"] > band["estimate"]).all()


def test_one_cluster_of_reference_units_warns(lowest_panel):
    with pytest.warns(UserWarning, match="all the comparison units are in one cluster"):
        fit = did_continuous("y ~ 1", lowest_panel, **LOWEST_KW, dose="dose", cluster="major")
    assert fit.diagnostics["fewest_comparison_clusters"] == 1
    assert "give reference_dose" in fit.diagnostics["warnings"][-1]


def test_clustered_fit_does_not_depend_on_threads_or_batches(lowest_panel):
    options = dict(
        LOWEST_KW,
        dose="dose",
        reference_dose=3.0,
        cluster="major",
        multiplier_bootstrap_reps=40,
        multiplier_bootstrap_seed=2,
    )
    one = did_continuous("y ~ 1", lowest_panel, **(options | dict(n_threads=1)))
    many = did_continuous("y ~ 1", lowest_panel, **(options | dict(n_threads=4, batch_units=61)))
    assert_allclose(many.vcov, one.vcov, rtol=1e-10)
    a, b = one.dose_response(uniform=True), many.dose_response(uniform=True)
    assert_allclose(b["ci_low"].to_numpy(), a["ci_low"].to_numpy(), rtol=1e-10)


@pytest.mark.parametrize(
    "options,message",
    [
        (dict(cluster="id"), "above the unit"),
        (dict(cluster="nowhere"), "unknown cluster columns"),
        (dict(cluster="x"), "constant within each unit"),
    ],
)
def test_cluster_errors(lowest_panel, options, message):
    with pytest.raises(ValueError, match=message):
        did_continuous(
            "y ~ 1", lowest_panel, **LOWEST_KW, dose="dose", reference_dose=3.0, **options
        )


# Several fixed-effect terms (experimental): the comparison regressions are fitted by
# hdfe-stream, and must equal the same regressions with all but one term as dummies,
# and the dense calculation with every term as dummies.


def _with_industry(panel):
    """A unit-level industry (not nested in region) and macro region (nested in it)."""
    return panel.with_columns(industry=(pl.col("id") * 7 + 3) % 5, macro=pl.col("region") // 3)


@pytest.mark.parametrize("cg", CONTROLS)
@pytest.mark.parametrize("kind", COVARIATE_FITS)
def test_several_fixed_effects_equal_one_with_the_others_as_dummies(covariate_panel, kind, cg):
    fml, options = COVARIATE_FITS[kind]
    data = _with_industry(covariate_panel)
    options = dict(**COVARIATE_KW, **options, control_group=cg)
    with pytest.warns(ExperimentalWarning, match="covariates and fixed effects goes beyond"):
        absorbed = did_continuous(fml + " + industry", data, **options)
    dummies = fit_quietly(fml.replace("| region", "+ C(industry) | region"), data, **options)
    assert absorbed.diagnostics["fixed_effect"] == "region + industry"
    assert absorbed.diagnostics["fixed_effect_levels"] == [8, 5]
    assert absorbed.diagnostics["covariate_names"] == ["x"]
    assert absorbed._summary_lines()[-1] == (
        "covariates: x; fixed effects: region + industry (experimental: see the docs)"
    )
    assert_allclose(absorbed.beta, dummies.beta, rtol=1e-10)
    assert_allclose(absorbed.vcov, dummies.vcov, rtol=1e-8)
    assert_allclose(absorbed._att.beta, dummies._att.beta, rtol=1e-9, atol=1e-12)
    assert_allclose(absorbed._att.vcov, dummies._att.vcov, rtol=1e-8, atol=1e-15)
    for target in ("level", "slope"):
        a, b = absorbed.dose_response(target), dummies.dose_response(target)
        assert_allclose(a["estimate"].to_numpy(), b["estimate"].to_numpy(), rtol=1e-9, atol=1e-12)
        assert_allclose(a["std_error"].to_numpy(), b["std_error"].to_numpy(), rtol=1e-8)
    a, b = absorbed.aggregate(), dummies.aggregate()
    assert_allclose(a.beta, b.beta, rtol=1e-9, atol=1e-12)
    assert_allclose(a.se, b.se, rtol=1e-8)


def test_a_nested_fixed_effect_changes_nothing(covariate_panel):
    """Macro regions are nested in regions, so region + macro spans what region does:
    the hdfe-stream fit must give the exact fit of region alone."""
    small = _with_industry(covariate_panel.filter(pl.col("id") < SMALL))
    options = dict(**COVARIATE_KW, dose="dose")
    nested = fit_quietly("y ~ x | region + macro", small, **options)
    exact = fit_quietly("y ~ x | region", small, **options)
    assert_allclose(nested.beta, exact.beta, rtol=1e-11)
    assert_allclose(nested.vcov, exact.vcov, rtol=1e-10)
    assert_allclose(
        nested.dose_response("slope")["std_error"],
        exact.dose_response("slope")["std_error"],
        rtol=1e-10,
    )


@pytest.mark.parametrize(
    "fml,covariates,cg",
    [
        ("y ~ x | region + industry", True, "never_treated"),
        ("y ~ 1 | region + industry", False, "not_yet_treated"),
        ("y ~ x | region^macro + industry", True, "not_yet_treated"),
    ],
)
def test_several_fixed_effects_match_the_dense_oracle(covariate_panel, fml, covariates, cg):
    small = _with_industry(covariate_panel.filter(pl.col("id") < SMALL))
    fit = fit_quietly(fml, small, **COVARIATE_KW, dose="dose", control_group=cg)
    estimate, se = general_oracle(
        small, fit, "y", "dose", cg, covariates=covariates, fixed_effect=["region", "industry"]
    )
    assert_allclose(fit.beta, estimate, rtol=1e-10)
    assert_allclose(fit.se, se, rtol=1e-8)


def test_several_fixed_effects_unbalanced_weighted_and_clustered(covariate_panel):
    """The hdfe-stream path with every other extension: an unbalanced panel, weights
    and clustering above the unit."""
    small = _unbalanced(_with_industry(covariate_panel.filter(pl.col("id") < SMALL)), 5)
    small = small.with_columns(w=1 + (pl.col("id") % 4) / 4, cl=pl.col("id") % 40)
    fit = fit_quietly(
        "y ~ x | region + industry",
        small,
        **COVARIATE_KW,
        dose="dose",
        control_group="not_yet_treated",
        incomplete_units="keep",
        weights="w",
        cluster="cl",
    )
    estimate, se = general_oracle(
        small,
        fit,
        "y",
        "dose",
        "not_yet_treated",
        covariates=True,
        fixed_effect=["region", "industry"],
        weights="w",
        cluster="cl",
    )
    assert_allclose(fit.beta, estimate, rtol=1e-10)
    assert_allclose(fit.se, se, rtol=1e-8)


def test_lowest_dose_with_several_fixed_effects_matches_the_dense_oracle(lowest_panel):
    data = _with_industry(lowest_panel)
    fit = fit_quietly(
        "y ~ 1 | region + industry", data, **LOWEST_KW, dose="dose", reference_dose=3.0
    )
    estimate, se = general_oracle(
        data,
        fit,
        "y",
        "dose",
        "lowest_dose",
        reference=lambda d: d <= 3.0,
        fixed_effect=["region", "industry"],
    )
    assert_allclose(fit.beta, estimate, rtol=1e-10)
    assert_allclose(fit.se, se, rtol=1e-8)


@pytest.mark.parametrize("cg", CONTROLS)
def test_levels_not_linked_to_the_comparison_units_are_left_out(covariate_panel, cg):
    """The never-treated units of region 7 are all in industry 9, which no other region
    has: region 7 and industry 9 form a component of their own. A treated unit of region
    7 in industry 9 can be adjusted; one of region 7 in another industry cannot, though
    each of its levels has comparison units."""
    small = _with_industry(covariate_panel.filter(pl.col("id") < SMALL))
    linked = (pl.col("region") == 7) & ((pl.col("first_treat") == 0) | (pl.col("id") % 2 == 0))
    data = small.with_columns(
        pl.when(linked).then(9).otherwise(pl.col("industry")).alias("industry")
    )
    fit = fit_quietly(
        "y ~ x | region + industry", data, **COVARIATE_KW, dose="dose", control_group=cg
    )
    left = fit.diagnostics["treated_units_left_out"]
    unlinked = data.filter(
        (pl.col("time") == 1)
        & (pl.col("region") == 7)
        & (pl.col("first_treat") > 0)
        & (pl.col("industry") != 9)
    )
    if cg == "never_treated":
        expected = unlinked.group_by("first_treat").len()
        assert left == dict(zip(expected["first_treat"], expected["len"], strict=True))
    else:
        # later cohorts' units of region 7 link it to the other industries in the
        # earlier cells, but every cohort has cells compared with the never treated alone
        assert 0 < sum(left.values()) <= unlinked.height
    assert any("cannot be estimated" in w for w in fit.diagnostics["warnings"])
    estimate, se = general_oracle(
        data, fit, "y", "dose", cg, covariates=True, fixed_effect=["region", "industry"]
    )
    assert_allclose(fit.beta, estimate, rtol=1e-10)
    assert_allclose(fit.se, se, rtol=1e-8)


@pytest.mark.resource
def test_several_fixed_effects_under_a_memory_limit(
    resource_runner, resource_enforcement, covariate_panel, tmp_path
):
    """Under a memory watchdog, with small staging pieces and batches, the same fit."""
    import os
    import sys

    options = dict(preprocess_mb=4, batch_units=37, n_threads=2)
    code = f"""
import json
import warnings
import polars as pl
from benchmarks._files import read_csv
from diff_stream import did_continuous
warnings.simplefilter("ignore")
data = read_csv({str(DATA / "continuous_covariates_panel.csv")!r}).with_columns(
    industry=(pl.col("id") * 7 + 3) % 5)
r = did_continuous("y ~ x | region + industry", data, panel=("id", "time"),
    cohort="first_treat", dose="dose", workdir={str(tmp_path / "child")!r}, **{options!r})
print(json.dumps(dict(beta=r.beta.tolist(), vcov=r.vcov.tolist(),
    curve=r.dose_response()["std_error"].to_list())))
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
    result = resource_runner([sys.executable, "-c", code], memory_gb=1.5, env=env, timeout=300)
    assert result.status == "ok", result.as_dict()
    assert result.mode == resource_enforcement["mode"]
    fitted = json.loads(result.stdout.strip().splitlines()[-1])
    baseline = fit_quietly(
        "y ~ x | region + industry", _with_industry(covariate_panel), **COVARIATE_KW, dose="dose"
    )
    assert_allclose(fitted["beta"], baseline.beta, rtol=1e-10)
    assert_allclose(fitted["vcov"], baseline.vcov, rtol=1e-8)
    assert_allclose(fitted["curve"], baseline.dose_response()["std_error"], rtol=1e-8)


FIXED_EFFECTS = DATA / "continuous_fixed_effects_reference"


@pytest.mark.parametrize("cg", CONTROLS)
@pytest.mark.parametrize("kind", COVARIATE_FITS)
def test_several_fixed_effects_match_diff_diff(covariate_panel, kind, cg):
    """diff-diff with both terms' levels as dummy covariates."""
    for name, h in json.loads((FIXED_EFFECTS / "PROVENANCE.json").read_text())["files"].items():
        assert digest(Path(__file__).parents[1] / name) == h, name
    fml, options = COVARIATE_FITS[kind]
    fit = fit_quietly(
        fml + " + industry",
        _with_industry(covariate_panel),
        **COVARIATE_KW,
        **options,
        control_group=cg,
    )
    name = f"{kind}_{cg}"
    overall = read_csv(FIXED_EFFECTS / f"overall_{name}.csv")
    assert_allclose(fit.beta, overall["estimate"].to_numpy(), rtol=1e-12)
    curves = read_csv(FIXED_EFFECTS / f"curves_{name}.csv")
    dose = COVARIATE_GRID if kind == "continuous" else None
    level, slope = fit.dose_response(dose=dose), fit.dose_response("slope", dose=dose)
    assert_allclose(level["dose"].to_numpy(), curves["dose"].to_numpy())
    for frame, target in ((level, "att"), (slope, "acrt")):
        assert_allclose(frame["estimate"].to_numpy(), curves[target].to_numpy(), rtol=1e-11)
        assert_allclose(
            frame["std_error"].to_numpy(), curves[f"{target}_se"].to_numpy(), rtol=1e-10
        )
    cells = read_csv(FIXED_EFFECTS / f"cells_{name}.csv")
    assert fit._att.cells == list(zip(cells["g"], cells["t"], strict=True))
    assert_allclose(fit._att.beta, cells["att"].to_numpy(), rtol=1e-10, atol=1e-13)
    assert_allclose(fit._acrt.beta, cells["acrt"].to_numpy(), rtol=1e-10, atol=1e-13)
    event = read_csv(FIXED_EFFECTS / f"event_{name}.csv")
    assert_allclose(fit.aggregate().beta, event["estimate"].to_numpy(), rtol=1e-10, atol=1e-13)


def test_several_fixed_effects_do_not_depend_on_threads_or_batches(covariate_panel):
    data = _with_industry(covariate_panel)
    options = dict(panel=("id", "time"), cohort="first_treat", dose="dose")
    base = fit_quietly("y ~ x | region + industry", data, **options, n_threads=2)
    one = fit_quietly(
        "y ~ x | region + industry", data, **options, n_threads=1, batch_units=37, preprocess_mb=4
    )
    assert_allclose(one.beta, base.beta, rtol=1e-11)
    assert_allclose(one.vcov, base.vcov, rtol=1e-9)
