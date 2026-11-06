"""Bounded weighted elimination of group intercepts and small Gram reduction.

Arrays contain one batch of whole groups. No treatment, time or panel policy
lives here; these helpers are candidates for hdfe_stream.core.
"""

import numpy as np
from scipy.linalg import solve_triangular


def center_groups(X, y, weights):
    """Eliminate intercepts using row weights (zero means excluded internally)."""
    mass = weights.sum(axis=1)
    if np.any(mass <= 0):
        raise ValueError("each group requires a positive-weight estimation observation")
    xm = row_sums(weights, X) / mass[:, None]
    ym = (weights * y).sum(axis=1) / mass
    return X - xm[:, None, :], y - ym[:, None], xm, ym


def moments(X, y, weights):
    xc, yc, _, _ = center_groups(X, y, weights)
    flat = xc.reshape(-1, xc.shape[2])
    weighted = weights.reshape(-1, 1) * flat
    return flat.T @ weighted, weighted.T @ yc.ravel()


# Batched sums over the period axis of (units, periods, columns) arrays, as
# matrix products: einsum would copy its operands and run single-threaded.


def row_sums(weights, X):
    """sum_t weights[b, t] * X[b, t, k] -> (units, columns)."""
    return np.matmul(weights[:, None, :], X)[:, 0, :]


def weighted_gram(X, weights):
    """sum_{b,t} weights[b, t] * X[b, t, i] * X[b, t, j] -> (columns, columns)."""
    flat = X.reshape(-1, X.shape[2])
    return flat.T @ (weights.reshape(-1, 1) * flat)


def weighted_squares(X, weights):
    """sum_{b,t} weights[b, t] * X[b, t, k] ** 2 -> (columns,)."""
    flat = X.reshape(-1, X.shape[2])
    return (flat * flat).T @ weights.ravel()


def independent_columns(gram, *, original_norm=None, tolerance=1e-10):
    """Select an ordered basis after scaling columns to unit Gram diagonal.

    Earlier columns have priority. Return aliases explicitly for the caller's
    diagnostics; consumers must separately protect identified target columns.

    A column is kept when its squared distance from the span of the columns kept
    before it exceeds `tolerance`: an ordered Cholesky factorization that grows
    one row per kept column (a triangular solve per column, O(K^3) in all).
    """
    diagonal = np.maximum(np.diag(gram), 0)
    scale = np.sqrt(diagonal)
    live = diagonal > 0 if original_norm is None else diagonal > original_norm * 1e-24
    safe = np.where(live, scale, 1)
    correlation = gram / safe[:, None] / safe[None, :]
    candidates = np.flatnonzero(live)
    factor = np.zeros((len(candidates), len(candidates)))  # lower triangular
    kept = []
    for j in candidates:
        m = len(kept)
        if m:
            row = solve_triangular(
                factor[:m, :m], correlation[j, kept], lower=True, check_finite=False
            )
            remaining = correlation[j, j] - row @ row
        else:
            row = np.zeros(0)
            remaining = correlation[j, j]
        if remaining > tolerance:
            factor[m, :m] = row
            factor[m, m] = np.sqrt(remaining)
            kept.append(j)
    return np.asarray(kept, dtype=int)
