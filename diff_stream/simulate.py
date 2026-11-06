"""Reproducible heterogeneous staggered-adoption panels."""

import numpy as np
import polars as pl


def simulate_panel(n_units=1000, periods=8, *, seed=42, noise=1.0):
    """Balanced panel with never-treated units, serial dependence and effects
    varying by cohort and exposure. Returns id, time, first_treat, y, truth.
    """
    if n_units < 10 or periods < 3:
        raise ValueError("use at least 10 units and 3 periods")
    rng = np.random.default_rng(seed)
    g = rng.choice(np.r_[0, np.arange(2, periods + 1)], n_units)
    g[:2] = [0, 2]
    t = np.arange(1, periods + 1)
    tau = np.where(
        (g[:, None] > 0) & (t >= g[:, None]), 0.5 + 0.15 * g[:, None] + 0.25 * (t - g[:, None]), 0
    )
    err = rng.normal(size=(n_units, periods))
    for j in range(1, periods):
        err[:, j] += 0.4 * err[:, j - 1]
    y = rng.normal(size=(n_units, 1)) + 0.1 * t + tau + noise * err
    return pl.DataFrame(
        {
            "id": np.repeat(np.arange(n_units), periods),
            "time": np.tile(t, n_units),
            "first_treat": np.repeat(g, periods),
            "y": y.ravel(),
            "truth": tau.ravel(),
        }
    )


def simulate_conditional_panel(n_units=1200, periods=5, *, seed=871, time_varying=True):
    """Confounded staggered panel with numeric and categorical covariates.

    Cohorts depend on unit-level x/z; effects differ by cohort and exposure.
    Time-varying x exercises the earlier-comparison-period convention.
    """
    if n_units < 30 or periods < 4:
        raise ValueError("use at least 30 units and four periods")
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n_units)
    z = rng.integers(0, 2, size=n_units)
    category = rng.choice(["a", "b", "c"], size=n_units)
    choices = np.array([0, periods - 2, periods - 1, periods])
    logits = np.column_stack(
        [np.zeros(n_units), 0.35 * x0 - 0.2 * z, -0.3 * x0 + 0.3 * z, 0.2 * x0 + 0.15 * z]
    )
    prob = np.exp(logits - logits.max(axis=1, keepdims=True))
    prob /= prob.sum(axis=1, keepdims=True)
    g = choices[(rng.random(n_units)[:, None] > prob.cumsum(axis=1)).sum(axis=1)]
    g[:4] = choices
    t = np.arange(1, periods + 1)
    x = np.repeat(x0[:, None], periods, axis=1)
    if time_varying:
        x += 0.1 * t + 0.1 * rng.normal(size=x.shape)
    tau = np.where(
        (g[:, None] > 0) & (t >= g[:, None]), 0.5 + 0.1 * g[:, None] + 0.2 * (t - g[:, None]), 0
    )
    y0 = (
        rng.normal(size=(n_units, 1))
        + 0.2 * t
        + 0.3 * t * x0[:, None]
        + 0.15 * t * z[:, None]
        + 0.4 * x
    )
    y = y0 + tau + rng.normal(size=x.shape)
    return pl.DataFrame(
        {
            "id": np.repeat(np.arange(n_units), periods),
            "time": np.tile(t, n_units),
            "first_treat": np.repeat(g, periods),
            "y": y.ravel(),
            "x": x.ravel(),
            "z": np.repeat(z, periods),
            "category": np.repeat(category, periods),
        }
    )
