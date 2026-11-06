"""Bounded weighted elimination of group intercepts and small Gram reduction.

Arrays contain one batch of whole groups. No treatment, time or panel policy
lives here; these helpers are candidates for hdfe_stream.core.
"""

import numpy as np


def center_groups(X, y, weights):
    """Eliminate intercepts using row weights (zero means excluded internally)."""
    mass = weights.sum(axis=1)
    if np.any(mass <= 0):
        raise ValueError("each group requires a positive-weight estimation observation")
    xm = np.einsum("bt,btk->bk", weights, X) / mass[:, None]
    ym = np.einsum("bt,bt->b", weights, y) / mass
    return X - xm[:, None, :], y - ym[:, None], xm, ym


def moments(X, y, weights):
    xc, yc, _, _ = center_groups(X, y, weights)
    return np.einsum("bti,bt,btj->ij", xc, weights, xc, optimize=True), np.einsum(
        "bti,bt,bt->i", xc, weights, yc, optimize=True
    )


def independent_columns(gram, *, original_norm=None, tolerance=1e-10):
    """Select an ordered basis after scaling columns to unit Gram diagonal.

    Earlier columns have priority. Return aliases explicitly for the caller's
    diagnostics; consumers must separately protect identified target columns.
    """
    diagonal = np.maximum(np.diag(gram), 0)
    scale = np.sqrt(diagonal)
    live = diagonal > 0 if original_norm is None else diagonal > original_norm * 1e-24
    safe = np.where(live, scale, 1)
    correlation = gram / safe[:, None] / safe[None, :]
    kept, roots = [], []
    for j in range(len(scale)):
        if not live[j]:
            continue
        if kept:
            cross = correlation[j, kept]
            root = np.linalg.solve(np.asarray(roots), cross)
            remaining = correlation[j, j] - root @ root
        else:
            root = np.zeros(0)
            remaining = correlation[j, j]
        if remaining > tolerance:
            for row in roots:
                row.append(0.0)
            roots.append([*root, float(np.sqrt(remaining))])
            kept.append(j)
    return np.asarray(kept, dtype=int)
