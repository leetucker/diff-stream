"""Small, intentionally dense independent regression references. Never production code."""

import numpy as np


def regression(df, method, cells, anticipation=0):
    ids = df["id"].to_numpy()
    times = df["time"].to_numpy()
    groups = df["first_treat"].to_numpy()
    y = df["y"].to_numpy()
    N = len(np.unique(ids))
    # Explicit unit dummies: no within transformation or production primitives.
    X = np.column_stack(
        [ids == u for u in np.unique(ids)] + [times == t for t in np.unique(times)[1:]]
    ).astype(float)
    treated = (groups != 0) & (times >= groups - anticipation)
    if method == "sunab":
        X = np.column_stack([X] + [(groups == g) & (times == t) for g, t in cells])
        pinv = np.linalg.pinv(X)
        b = pinv @ y
        W = pinv[-len(cells) :]
        residual = y - X @ b
        att = b[-len(cells) :]
    else:
        pinv = np.linalg.pinv(X[~treated])
        nuisance = pinv @ y[~treated]
        residual = y - X @ nuisance
        W = np.zeros((len(cells), len(y)))
        att = []
        for j, (g, t) in enumerate(cells):
            target = (groups == g) & (times == t)
            W[j, target] = 1 / target.sum()
            W[j, ~treated] -= X[target].mean(axis=0) @ pinv
            a = residual[target].mean()
            att.append(a)
            residual[target] -= a
        att = np.array(att)
    scores = np.column_stack([W[:, ids == u] @ residual[ids == u] for u in range(N)])
    return att, scores @ scores.T


def cs(df, cells, control_group, base_period):
    y = df["y"].to_numpy().reshape(-1, df["time"].n_unique())
    groups = df["first_treat"].to_numpy().reshape(y.shape)[:, 0]
    scores = np.zeros((len(y), len(cells)))
    beta = []
    for j, (g, t) in enumerate(cells):
        b = t - 1 if base_period == "varying" and t < g else g - 1
        d = groups == g
        c = (
            (groups == 0) | ((groups > max(t, b)) & (groups != g))
            if control_group == "notyettreated"
            else groups == 0
        )
        dy = y[:, t - 1] - y[:, b - 1]
        a, z = dy[d].mean(), dy[c].mean()
        beta.append(a - z)
        scores[d, j] = (dy[d] - a) / d.sum()
        scores[c, j] = -(dy[c] - z) / c.sum()
    return np.array(beta), scores
