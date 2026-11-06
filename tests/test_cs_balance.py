"""Independent weighted moments/logit checks for CS comparison balance."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import root
from scipy.special import expit

from benchmarks._estimators import fit_estimator, public
from benchmarks._files import read_csv

DATA = Path(__file__).parent / "data"
KW = dict(
    panel=("id", "time"),
    cohort="first_treat",
    n_threads=2,
    preprocess_mb=4,
    batch_units=37,
    effective_sample_warning=None,
)


def independent_rows(d, panel, target, control):
    base = 4 if target >= 5 else target - 1
    groups = [0] + (
        [7]
        if control == "notyettreated" and target == 6
        else [6, 7]
        if control == "notyettreated" and target == 3
        else []
    )
    select = d.filter(pl.col("first_treat").is_in([5, *groups]))
    if panel == "balanced":
        select = select.filter(pl.col("time") == min(target, base)).sort("id")
        post = np.ones(len(select), dtype=bool)
    else:
        select = select.filter(pl.col("time").is_in([base, target])).sort("id", "time")
        post = select["time"].to_numpy() == target
    X = np.column_stack([np.ones(len(select)), select["x"].to_numpy(), select["z"].to_numpy()])
    d1 = select["first_treat"].to_numpy() == 5
    w = select["w"].to_numpy()

    def score(b):
        return X.T @ (w * (d1 - expit(X @ b)))

    sol = root(
        score,
        np.zeros(3),
        jac=lambda b: -(X.T @ (X * (w * expit(X @ b) * (1 - expit(X @ b)))[:, None])),
        tol=1e-11,
    )
    assert np.max(abs(score(sol.x))) < 1e-7
    p = np.minimum(expit(X @ sol.x), 1 - 1e-6)
    after = w * np.where(d1, 1, np.where(p < 0.65, p / (1 - p), 0))
    return X, d1, w, after, post, base


def moments(x, w):
    w = w / w.sum()
    mean = w @ x
    variance = (
        mean * (1 - mean)
        if set(np.unique(x)) <= {0.0, 1.0}
        else (w @ ((x - mean) ** 2)) / (1 - w @ w)
    )
    return mean, variance, 1 / (w @ w)


@pytest.mark.parametrize("panel", ["balanced", "unbalanced"])
@pytest.mark.parametrize("target", [3, 6])
@pytest.mark.parametrize("control", ["nevertreated", "notyettreated"])
@pytest.mark.parametrize("method", ["reg", "ipw", "dr"])
def test_balance_selection_and_propensity_weights(panel, target, control, method, tmp_path):
    d = read_csv(
        DATA / ("richer_panel.csv" if panel == "balanced" else "richer_unbalanced_panel.csv")
    )
    ecdf = method == "dr" and target == 6 and control == "nevertreated"
    fit = fit_estimator(
        "y~x+z",
        d,
        **KW,
        incomplete_units=public("incomplete_units", panel),
        control_group=public("control_group", control),
        adjustment=public("adjustment", method),
        weights="w",
        allow_time_varying_weights=True,
        propensity_trim=0.65,
        covariate_balance=True,
        balance_distributions=ecdf,
        workdir=tmp_path,
    )
    table = fit.covariate_balance(cohort=5, time=target)
    X, d1, w, after, post, base = independent_rows(d, panel, target, control)
    assert set(table["stage"]) == ({"before"} if method == "reg" else {"before", "after"})
    assert set(table["term"]) == {"x", "z"}
    for row in table.iter_rows(named=True):
        mask = (
            np.ones(len(d1), dtype=bool)
            if panel == "balanced"
            else post == (row["stratum"] == "post")
        )
        x = X[mask, 1 if row["term"] == "x" else 2]
        dd = d1[mask]
        sw = w[mask] if row["stage"] == "before" else after[mask]
        m1, v1, e1 = moments(x[dd], sw[dd])
        m0, v0, e0 = moments(x[~dd], sw[~dd])
        _, b1, _ = moments(x[dd], w[mask][dd])
        _, b0, _ = moments(x[~dd], w[mask][~dd])
        smd = (
            (m1 - m0) / np.sqrt((b1 + b0) / 2)
            if abs(m1 - m0) > np.sqrt(np.finfo(float).eps)
            else m1 - m0
        )
        assert_allclose(
            [
                row["mean1"],
                row["mean0"],
                row["var1"],
                row["var0"],
                row["smd"],
                row["ess_treated"],
                row["ess_controls"],
            ],
            [m1, m0, v1, v0, smd, e1, e0],
            atol=1e-9,
            rtol=1e-10,
        )
        assert row["n_treated"] == dd.sum()
        assert row["n_controls"] == (~dd).sum()
        if row["term"] == "z":
            assert row["var_ratio"] is None
        if ecdf:
            grid = np.unique(x)
            gaps = [
                abs(
                    sw[dd & (x <= v)].sum() / sw[dd].sum()
                    - sw[(~dd) & (x <= v)].sum() / sw[~dd].sum()
                )
                for v in grid
            ]
            assert_allclose(
                [row["ecdf_mean"], row["ecdf_max"]],
                [abs(m1 - m0), abs(m1 - m0)] if row["term"] == "z" else [np.mean(gaps), max(gaps)],
                atol=1e-9,
                rtol=0,
            )
        elif row["term"] == "x":
            assert np.isnan(row["ecdf_max"])
        assert row["covariate_time"] == (
            min(base, target)
            if panel == "balanced"
            else target
            if row["stratum"] == "post"
            else base
        )
    assert fit.diagnostics["balance_source_passes"] > 0
    assert not list(tmp_path.iterdir())
    assert fit.aggregate(horizons=[0]).covariate_balance()["time"].to_list() != []


def test_balance_is_opt_in_and_does_not_change_estimation(tmp_path):
    d = read_csv(DATA / "richer_unbalanced_panel.csv")
    kwargs = dict(
        **KW,
        incomplete_units="keep",
        weights="w",
        allow_time_varying_weights=True,
        workdir=tmp_path,
    )
    plain = fit_estimator("y~x+z", d, **kwargs)
    balanced = fit_estimator("y~x+z", d, covariate_balance=True, **kwargs)
    assert_allclose(plain.beta, balanced.beta, atol=1e-12, rtol=0)
    assert_allclose(plain.vcov, balanced.vcov, atol=1e-12, rtol=0)
    with pytest.raises(ValueError, match="balance=True"):
        plain.covariate_balance()
    for kwargs in [
        dict(estimator="sunab", covariate_balance=True),
        dict(balance_distributions=True),
        dict(covariate_balance="yes"),
    ]:
        with pytest.raises((ValueError, TypeError)):
            fit_estimator("y~x", d, **KW, **kwargs)
    with pytest.raises(ValueError, match="requires covariates"):
        fit_estimator("y~1", d, covariate_balance=True, incomplete_units="keep", **KW)
    assert not list(tmp_path.iterdir())
    table = balanced.covariate_balance()
    assert len(table) > 0
    selected = balanced.aggregate(horizons=[0]).covariate_balance()
    assert all(t == g for g, t in selected.select("cohort", "time").iter_rows())


@pytest.mark.resource
def test_reference_and_balance_constrained(resource_runner, resource_enforcement, tmp_path):
    path = DATA / "richer_unbalanced_panel.csv"
    work = tmp_path / "child"
    code = f"""import json,warnings,polars as pl
from diff_stream import ReferenceWindowWarning, did_callaway_santanna
warnings.filterwarnings('ignore',category=ReferenceWindowWarning)
d=pl.read_csv({str(path)!r}).filter(pl.col('first_treat')!=0)
r=did_callaway_santanna('y~x+z',d,panel=('id','time'),cohort='first_treat',incomplete_units='keep',weights='w',allow_time_varying_weights=True,covariate_balance=True,n_threads=2,preprocess_mb=4,batch_units=37,effective_sample_warning=None,workdir={str(work)!r})
print(json.dumps(dict(beta=r.beta.tolist(),vcov=r.vcov.tolist(),balance_rows=r.covariate_balance().height,last=r.diagnostics['reference_cohort'])))
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
    value = json.loads(result.stdout)
    assert value["last"] == 7 and value["balance_rows"] > 0
    assert not list(work.iterdir())
    if result.kernel:
        assert result.kernel["events"]["oom_kill"] == 0
