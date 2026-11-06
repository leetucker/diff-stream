"""Bounded streaming sums of cluster influence scores times multipliers.

The multiplier samplers are reused from hdfe-stream's wild bootstrap. Its OLS
restricted-fit/studentization engine is deliberately not used here. Scores may
belong to any asymptotically linear estimator; no DiD conventions live here.
"""

from numbers import Integral

import numpy as np
from hdfe_stream.core.wildboot import WEIGHTS
from scipy.special import ndtri

MAX_DRAW_ENTRIES = 1 << 18


def validate_bootstrap(reps, seed, weights):
    if (
        isinstance(reps, (bool, np.bool_))
        or not isinstance(reps, Integral)
        or (reps != 0 and reps < 2)
    ):
        raise ValueError("multiplier_bootstrap_reps must be 0 (disabled) or an integer >= 2")
    if seed is not None and (
        isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0
    ):
        raise ValueError("multiplier_bootstrap_seed must be a nonnegative integer or None")
    if not isinstance(weights, str) or weights not in WEIGHTS:
        raise ValueError(f"multiplier_bootstrap_weights must be one of {sorted(WEIGHTS)}")


class StreamingMultiplier:
    """Keep repetitions-by-score-width state, never clusters-by-repetitions.

    A unit receives the same multiplier in every score block. RNG draws are
    unit-major, so changing numerical batch size preserves assignments for a
    fixed unit traversal. Repartitioning can change traversal and assignments.
    """

    def __init__(self, width, reps, seed, weights):
        validate_bootstrap(reps, seed, weights)
        self.reps = int(reps)
        self.rng = np.random.default_rng(seed)
        self.sampler = WEIGHTS[weights]
        self.draws = np.zeros((self.reps, width))
        self.n_clusters = 0

    def update(self, *blocks):
        rows = len(blocks[0])
        if (
            any(len(b) != rows or b.ndim != 2 for b in blocks)
            or sum(b.shape[1] for b in blocks) != self.draws.shape[1]
        ):
            raise ValueError("multiplier score blocks have incompatible shapes")
        step = max(1, MAX_DRAW_ENTRIES // self.reps)
        for start in range(0, rows, step):
            stop = min(rows, start + step)
            xi = np.asarray(
                self.sampler((stop - start) * self.reps, self.rng), dtype=float
            ).reshape(stop - start, self.reps)
            column = 0
            for block in blocks:
                width = block.shape[1]
                self.draws[:, column : column + width] += xi.T @ block[start:stop]
                column += width
        self.n_clusters += rows


def multiplier_band(draws, alpha, *, joint=True, analytic_se=None):
    """IQR-studentized bands, using R did's empirical (type 1) quantiles.

    Optional analytic_se uses fixed analytic studentization instead. Draws
    are centered perturbations on the estimate scale, not resampled estimates.
    """
    if isinstance(alpha, (bool, np.bool_)) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not isinstance(joint, bool):
        raise ValueError("joint must be boolean")
    if analytic_se is None:
        q = np.quantile(draws, [0.25, 0.75], axis=0, method="inverted_cdf")
        se = (q[1] - q[0]) / (ndtri(0.75) - ndtri(0.25))
    else:
        se = np.asarray(analytic_se)
    if not np.isfinite(draws).all() or not np.isfinite(se).all() or np.any(se <= 0):
        raise ValueError(
            "selected coefficients have zero or non-finite bootstrap/analytic scale; select supported coefficients or increase multiplier_bootstrap_reps"
        )
    t = np.abs(draws / se)
    critical = np.quantile(t.max(axis=1) if joint else t, 1 - alpha, axis=0, method="inverted_cdf")
    return se, critical
