"""Sampling-weight validation and optional weighted batch adapters.

No treatment or panel conventions live here. Consumer code owns restrictions
such as weights being constant within a sampling unit.
"""

import numpy as np


def validate_weights(values):
    """Require finite, strictly positive numeric sampling weights."""
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("sampling weights must be finite and strictly positive")


def weighted_system(parts):
    """Accept (X, d, Y[, weights]); preserve the unweighted consumer protocol."""
    X, d, Y = parts[:3]
    w = parts[3] if len(parts) == 4 else np.ones(len(d))
    return X, d, Y, w


def weighted_linear(parts):
    """Accept (X, y[, weights]) for a weighted least-squares stratum."""
    X, y = parts[:2]
    return X, y, parts[2] if len(parts) == 3 else np.ones(len(y))
