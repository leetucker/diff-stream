"""Shared-pass binary MLE and multi-outcome control-regression systems.

No panel or treatment-cohort conventions live here. A transform selects a
system's sample and returns (design, binary group, outcome matrix[, weights]) per batch.
Intended for eventual hdfe_stream.core; current models are unpenalized, no FE.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from .weights import weighted_linear, weighted_system


def solve_checked(matrix, rhs, context):
    """Solve in diagonally scaled coordinates; never silently drop columns."""
    scale = np.sqrt(np.maximum(np.diag(matrix), 0))
    if np.any(scale <= 0) or not np.isfinite(matrix).all():
        raise ValueError(f"{context}: singular nuisance design")
    normalized = matrix / np.outer(scale, scale)
    if np.linalg.eigvalsh(normalized)[0] <= 1e-12:
        raise ValueError(f"{context}: singular or ill-conditioned nuisance design")
    vector = np.ndim(rhs) == 1
    target = np.asarray(rhs) / (scale if vector else scale[:, None])
    answer = np.linalg.solve(normalized, target)
    return answer / (scale if vector else scale[:, None])


def fit_linear_systems(batches, transforms, width):
    """Fit weighted OLS on arbitrary row strata together in one source pass.

    Transforms return (X, y[, weights]). Return coefficient and Gram pairs, so callers can
    propagate estimation uncertainty through arbitrary smooth functionals.
    """
    if not transforms:
        return []
    grams = [np.zeros((width, width)) for _ in transforms]
    rhs = [np.zeros(width) for _ in transforms]
    counts = np.zeros(len(transforms), dtype=np.int64)
    for raw in batches():
        for j, transform in enumerate(transforms):
            X, y, w = weighted_linear(transform(raw))
            counts[j] += len(y)
            grams[j] += X.T @ (w[:, None] * X)
            rhs[j] += X.T @ (w * y)
    return [
        (
            solve_checked(
                Q,
                q,
                f"stratified outcome regression (system {j}, {counts[j]} observations, {width} columns)",
            ),
            Q,
        )
        for j, (Q, q) in enumerate(zip(grams, rhs, strict=True))
    ]


@dataclass
class NuisanceSystem:
    beta: np.ndarray
    regression: np.ndarray
    control_gram: np.ndarray
    n_treated: int = 0
    n_controls: int = 0
    iterations: int = 0
    treated_weight: float = 0.0
    control_weight: float = 0.0
    control_weight_squares: float = 0.0
    treated_weight_squares: float = 0.0
    design_scale: np.ndarray | None = None


def fit_systems(batches, transforms, method, names, *, tol=1e-10, maxiter=100):
    """Fit independent systems together, scanning the source once per Newton step.

    Multi-outcome control OLS systems share a Gram solve. Logistic MLE uses
    damped Newton steps, scaled solves and an explicit separation check.
    """
    K = len(names)
    systems = [None] * len(transforms)
    passes = 0
    for raw in batches():
        for j, transform in enumerate(transforms):
            X, d, Y, w = weighted_system(transform(raw))
            if systems[j] is None:
                systems[j] = NuisanceSystem(
                    np.zeros(K),
                    np.zeros((K, Y.shape[1])),
                    np.zeros((K, K)),
                    design_scale=np.zeros(K),
                )
            s = systems[j]
            s.n_treated += int(d.sum())
            s.n_controls += int((~d).sum())
            s.treated_weight += float(w[d].sum())
            s.control_weight += float(w[~d].sum())
            s.control_weight_squares += float(w[~d] @ w[~d])
            s.treated_weight_squares += float(w[d] @ w[d])
            s.design_scale += np.einsum("i,ij,ij->j", w, X, X)
            if method != "ipw":
                xc = X[~d]
                s.control_gram += xc.T @ (w[~d, None] * xc)
                s.regression += xc.T @ (w[~d, None] * Y[~d])
    passes += 1
    for s in systems:
        if not s.n_treated or not s.n_controls:
            raise ValueError("both nuisance groups need observations")
        s.design_scale = np.sqrt(s.design_scale / (s.treated_weight + s.control_weight))
        if method != "ipw":
            s.regression = solve_checked(s.control_gram, s.regression, "outcome regression")
        if "Intercept" in names:
            s.beta[names.index("Intercept")] = np.log(s.treated_weight / s.control_weight)
    if method == "reg":
        return systems, passes

    def stats(betas, indices):
        nonlocal passes
        out = {j: [0.0, np.zeros(K), np.zeros((K, K)), np.inf] for j in indices}
        loss_correction = dict.fromkeys(indices, 0.0)
        for raw in batches():
            for j in indices:
                X, d, _, w = weighted_system(transforms[j](raw))
                if not len(d):
                    continue
                eta = X @ betas[j]
                p = expit(eta)
                # Compensated accumulation keeps the objective stable across
                # many small batches without retaining their individual sums.
                increment = float(w @ (np.logaddexp(0, eta) - d * eta)) - loss_correction[j]
                total = out[j][0] + increment
                loss_correction[j] = (total - out[j][0]) - increment
                out[j][0] = total
                out[j][1] += X.T @ (w * (d - p))
                out[j][2] += X.T @ ((w * p * (1 - p))[:, None] * X)
                out[j][3] = min(out[j][3], float(np.min((2 * d.astype(float) - 1) * eta)))
        passes += 1
        return out

    betas = [s.beta.copy() for s in systems]
    active = set(range(len(systems)))
    current = stats(betas, active)
    for _iteration in range(maxiter):
        steps, done = {}, set()
        for j in active:
            loss, gradient, H, margin = current[j]
            if margin > 1e-8:
                raise ValueError("propensity logit: perfect separation / no overlap")
            step = solve_checked(H, gradient, "propensity logit")
            # Coefficients have inverse covariate units. Test each coordinate
            # in weighted RMS design units so changing meters to micrometers
            # cannot turn an unfinished fit into a converged one. The scale is
            # fixed from the first pass, rather than shrinking with p*(1-p)
            # near separation. No extra source pass or row storage is needed.
            scale = systems[j].design_scale
            if np.all(np.abs(step) * scale <= tol * (1 + np.abs(betas[j]) * scale)):
                done.add(j)
            else:
                steps[j] = step
        active -= done
        if not active:
            for j, s in enumerate(systems):
                s.beta = betas[j]
            return systems, passes
        candidates = list(betas)
        for j in active:
            candidates[j] = betas[j] + steps[j]
        trial = stats(candidates, active)
        for _backtrack in range(30):
            # The objective is a SUM, so its floating-point resolution grows
            # with sample size. Near the optimum, a genuine Newton improvement
            # can be smaller than summation noise. An absolute-only allowance
            # causes repeated tiny backtracks/full source scans at large N.
            # Keep the coefficient-step convergence test unchanged.
            bad = {
                j
                for j in active
                if not np.isfinite(trial[j][0])
                or trial[j][0]
                > current[j][0]
                + max(1e-10, 64 * np.finfo(float).eps * max(1.0, abs(current[j][0])))
            }
            if not bad:
                break
            for j in bad:
                steps[j] *= 0.5
                candidates[j] = betas[j] + steps[j]
            trial.update(stats(candidates, bad))
        else:
            raise ValueError(
                "propensity logit line search failed; check overlap and covariate scaling"
            )
        for j in active:
            betas[j] = candidates[j]
            current[j] = trial[j]
            systems[j].iterations += 1
    raise ValueError("propensity logit did not converge; possible quasi-separation / poor overlap")
