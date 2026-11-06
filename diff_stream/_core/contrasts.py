"""Reusable ATT mean contrasts with regression/estimated-propensity corrections.

Outcomes can be any columns, including changes supplied by a DiD consumer.
Scores are on the estimate scale, so joint covariance is sum_i phi_i phi_i'.
The array path supplements the shared expression-based IPW implementation.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl
from hdfe_stream.core.ipw import ipw_weight
from scipy.special import expit

from .nuisance import solve_checked
from .weights import weighted_system

_ATT_WEIGHT = ipw_weight("d", pl.col("p"), "att")


def att_parts(X, d, beta, control_max):
    """R-compatible ATT convention: cap p at 1-1e-6, trim controls only.

    The propensity and regression models still use all selected units.
    Trim membership is held fixed for analytic inference.
    """
    p = np.minimum(expit(X @ beta), 1 - 1e-6)
    raw = pl.DataFrame({"d": d, "p": p}).select(_ATT_WEIGHT.alias("w"))["w"].to_numpy()
    control = np.where((~d) & (p < control_max), raw, 0.0)
    return p, control


@dataclass
class AdjustedContrast:
    nuisance: object
    method: str
    control_max: float
    treated_total: float
    control_total: float
    treated_mean: np.ndarray
    control_mean: np.ndarray
    outcome_adjustment: np.ndarray
    propensity_adjustment: np.ndarray
    diagnostics: dict

    @property
    def estimate(self):
        return self.treated_mean - self.control_mean

    def scores(self, X, d, Y, weights=None):
        if not len(d):
            return np.zeros_like(Y)
        if self.method == "reg":
            p, wc = np.zeros(len(d)), np.zeros(len(d))
        else:
            p, wc = att_parts(X, d, self.nuisance.beta, self.control_max)
        residual = Y if self.method == "ipw" else Y - X @ self.nuisance.regression
        phi = d[:, None] * (residual - self.treated_mean) / self.treated_total
        if self.method != "reg":
            phi -= wc[:, None] * (residual - self.control_mean) / self.control_total
            phi += ((d - p)[:, None] * X) @ self.propensity_adjustment
        if self.method != "ipw":
            phi += (~d)[:, None] * residual * (X @ self.outcome_adjustment)[:, None]
        return phi if weights is None else phi * weights[:, None]


def fit_contrasts(batches, transforms, systems, method, control_max):
    """One common moment pass after nuisance fitting, for any number of outcomes."""
    moments = []
    for s in systems:
        K, L = s.regression.shape
        moments.append(
            dict(
                w0=0.0,
                w02=0.0,
                s0=np.zeros(L),
                s1=np.zeros(L),
                xt=np.zeros(K),
                xc=np.zeros(K),
                xcy=np.zeros((K, L)),
                H=np.zeros((K, K)),
                pmin=1.0,
                pmax=0.0,
                trimmed=0,
                capped=0,
            )
        )
    for raw in batches():
        for j, transform in enumerate(transforms):
            X, d, Y, w = weighted_system(transform(raw))
            if not len(d):
                continue
            s, m = systems[j], moments[j]
            residual = Y if method == "ipw" else Y - X @ s.regression
            m["s1"] += w[d] @ residual[d]
            m["xt"] += w[d] @ X[d]
            if method != "reg":
                p, wc = att_parts(X, d, s.beta, control_max)
                wc = wc * w
                m["pmin"] = min(m["pmin"], float(p.min()))
                m["pmax"] = max(m["pmax"], float(p.max()))
                m["trimmed"] += int(np.sum((~d) & (p >= control_max)))
                m["capped"] += int(np.sum(p >= 1 - 1e-6))
                m["w0"] += wc.sum()
                m["w02"] += wc @ wc
                m["s0"] += wc @ residual
                m["xc"] += wc @ X
                m["xcy"] += X.T @ (wc[:, None] * residual)
                m["H"] += X.T @ ((w * p * (1 - p))[:, None] * X)
    fitted = []
    for s, m in zip(systems, moments, strict=True):
        wt, wc = s.treated_weight, m["w0"]
        if method != "reg" and wc <= 0:
            raise ValueError("no controls remain after propensity trimming")
        mt = m["s1"] / wt
        mc = m["s0"] / wc if method != "reg" else np.zeros_like(mt)
        jo = -m["xt"] / wt
        if method == "dr":
            jo += m["xc"] / wc
        ao = (
            solve_checked(s.control_gram, jo, "outcome regression")
            if method != "ipw"
            else np.zeros(len(jo))
        )
        jp = (
            -(m["xcy"] - np.outer(m["xc"], mc)) / wc if method != "reg" else np.zeros_like(m["xcy"])
        )
        ap = solve_checked(m["H"], jp, "propensity information") if method != "reg" else jp
        fitted.append(
            AdjustedContrast(
                s,
                method,
                control_max,
                wt,
                wc,
                mt,
                mc,
                ao,
                ap,
                dict(
                    n_treated=s.n_treated,
                    n_controls=s.n_controls,
                    effective_treated=s.treated_weight**2 / s.treated_weight_squares,
                    trimmed_controls=m["trimmed"],
                    capped_probabilities=m["capped"],
                    effective_controls=wc * wc / m["w02"]
                    if method != "reg"
                    else s.control_weight**2 / s.control_weight_squares,
                    propensity_min=m["pmin"] if method != "reg" else None,
                    propensity_max=m["pmax"] if method != "reg" else None,
                    iterations=s.iterations,
                ),
            )
        )
    return fitted
