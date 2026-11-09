"""Compiled passes of the two-stage (Gardner) fit and its first-stage reduction.

Each pass visits every unit and does a handful of small weighted sums over the
unit's periods and the P design columns (the period dummies and covariates).
In NumPy that is a few hundred array operations per batch of units, which the
interpreter's lock serializes. The kernels do the same sums in one compiled
call per batch.

Units are processed in fixed-size chunks, and a chunk adds its units in order,
so the sums, and with them the estimates, do not depend on the number of
threads; the chunks' partial sums are added in order afterwards. The NumPy code
in ``linear.py`` and ``gardner.py`` remains the reference; the tests compare
the two.
"""

import numpy as np
from numba import njit, prange

ENABLED = True  # tests switch it off to compare with the array path
CHUNK = 64  # units per chunk; fixed so results do not depend on threads


@njit(cache=True, inline="always")
def _design(x, i, t, T1, columns, out):
    """Row (unit i, period t) of the design restricted to ``columns``:
    period dummies for periods 1.. then covariates, with missing values as 0."""
    for a in range(columns.shape[0]):
        j = columns[a]
        if j < T1:
            out[t, a] = 1.0 if (t >= 1 and j == t - 1) else 0.0
        else:
            v = x[i, t, j - T1]
            out[t, a] = v if np.isfinite(v) else 0.0


@njit(parallel=True, cache=True)
def reduce_moments(y, x, w, wc, columns, do_clean, Q, q, norms, Qc, qc, bad):
    """Within-unit moments of all columns for the rows with weights ``w``, and
    (``Qc``, ``qc``) for the rows with weights ``wc``; ``norms`` are the columns'
    weighted sums of squares. ``bad[chunk]`` is 1 when a unit has no weight.

    A unit's demeaned design is nonzero only in the columns where one of its
    rows is, which with fixed-effect dummies is a few of the P (its own levels,
    not all of them), so when it is under half of them the products visit those
    columns alone. The other entries would add exact zeros."""
    n, T = y.shape
    P = columns.shape[0]
    T1 = T - 1
    chunks = (n + CHUNK - 1) // CHUNK
    for c in prange(chunks):
        X = np.empty((T, P))
        xm = np.zeros(P)
        yv = np.zeros(T)
        act = np.empty(P, dtype=np.int64)
        for i in range(c * CHUNK, min(n, (c + 1) * CHUNK)):
            for t in range(T):
                _design(x, i, t, T1, columns, X)
                yv[t] = y[i, t] if np.isfinite(y[i, t]) else 0.0
            nact = 0
            for a in range(P):
                for t in range(T):
                    if X[t, a] != 0.0:
                        act[nact] = a
                        nact += 1
                        break
            if 2 * nact >= P:
                # mostly dense: contiguous loops over every column are faster than
                # indexed ones, and the skipped entries are exact zeros either way
                nact = P
                for a in range(P):
                    act[a] = a
            for pass_ in range(2 if do_clean else 1):
                mass = 0.0
                for t in range(T):
                    mass += w[i, t] if pass_ == 0 else wc[i, t]
                if mass <= 0.0:
                    bad[c] = 1
                    continue
                for a in range(P):
                    xm[a] = 0.0
                ym = 0.0
                for t in range(T):
                    wt = w[i, t] if pass_ == 0 else wc[i, t]
                    ym += wt * yv[t]
                    for ia in range(nact):
                        a = act[ia]
                        xm[a] += wt * X[t, a]
                ym /= mass
                for ia in range(nact):
                    a = act[ia]
                    xm[a] /= mass
                for t in range(T):
                    wt = w[i, t] if pass_ == 0 else wc[i, t]
                    if wt == 0.0:
                        continue
                    ry = yv[t] - ym
                    for ia in range(nact):
                        a = act[ia]
                        xa = (X[t, a] - xm[a]) * wt
                        if pass_ == 0:
                            q[c, a] += xa * ry
                            if nact == P:
                                # every column is active: a contiguous loop vectorizes
                                for b in range(P):
                                    Q[c, a, b] += xa * (X[t, b] - xm[b])
                            else:
                                for ib in range(nact):
                                    b = act[ib]
                                    Q[c, a, b] += xa * (X[t, b] - xm[b])
                        else:
                            qc[c, a] += xa * ry
                            if nact == P:
                                for b in range(P):
                                    Qc[c, a, b] += xa * (X[t, b] - xm[b])
                            else:
                                for ib in range(nact):
                                    b = act[ib]
                                    Qc[c, a, b] += xa * (X[t, b] - xm[b])
            for t in range(T):
                for ia in range(nact):
                    a = act[ia]
                    norms[c, a] += w[i, t] * X[t, a] * X[t, a]


@njit(cache=True, inline="always")
def _first_stage(g, y, x, w, i, T1, columns, theta, times, anticipation, X, xm, u1, score1, obs):
    """One unit's first-stage residuals ``u1`` (clean-sample imputation), the
    clean-weighted mean ``xm`` of its design and its first-stage score.
    Returns 0, or 1 when the unit has no untreated observation."""
    T = times.shape[0]
    P = columns.shape[0]
    for t in range(T):
        _design(x, i, t, T1, columns, X)
    mass = 0.0
    for a in range(P):
        xm[a] = 0.0
    alpha = 0.0
    for t in range(T):
        obs[t] = np.isfinite(y[i, t])
        use = obs[t] and (g[i] == 0 or times[t] < g[i] - anticipation)
        cw = w[i, t] if use else 0.0
        fit = 0.0
        for a in range(P):
            fit += X[t, a] * theta[a]
        u1[t] = (y[i, t] - fit) if obs[t] else 0.0
        mass += cw
        alpha += cw * u1[t]
        for a in range(P):
            xm[a] += cw * X[t, a]
    if mass <= 0.0:
        return 1
    alpha /= mass
    for a in range(P):
        xm[a] /= mass
        score1[a] = 0.0
    for t in range(T):
        use = obs[t] and (g[i] == 0 or times[t] < g[i] - anticipation)
        cw = w[i, t] if use else 0.0
        u1[t] = (u1[t] - alpha) if obs[t] else 0.0
        for a in range(P):
            score1[a] += cw * u1[t] * (X[t, a] - xm[a])
    return 0


@njit(cache=True, inline="always")
def _indicator(g, i, t, times, anticipation, horizons, m, dynamic):
    """Second-stage indicator of coefficient m for unit i in period t (before
    the observed-row condition)."""
    if g[i] == 0:
        return 0.0
    event = times[t] - g[i]
    if dynamic:
        return 1.0 if event == horizons[m] else 0.0
    return 1.0 if event >= -anticipation else 0.0


@njit(parallel=True, cache=True)
def second_stage_sums(
    g, y, x, w, columns, theta, times, anticipation, horizons, dynamic, mass, rhs, C, bad
):
    """The second stage's sums: Z'w, Z'w u1 and Z'w (X - xm) (``gardner._fit``)."""
    n, T = y.shape
    P = columns.shape[0]
    M = mass.shape[1]
    T1 = T - 1
    chunks = (n + CHUNK - 1) // CHUNK
    for c in prange(chunks):
        X = np.empty((T, P))
        xm = np.zeros(P)
        u1 = np.zeros(T)
        s1 = np.zeros(P)
        obs = np.zeros(T, dtype=np.bool_)
        for i in range(c * CHUNK, min(n, (c + 1) * CHUNK)):
            if _first_stage(
                g, y, x, w, i, T1, columns, theta, times, anticipation, X, xm, u1, s1, obs
            ):
                bad[c] = 1
                continue
            for t in range(T):
                if not obs[t]:
                    continue
                for m in range(M):
                    z = _indicator(g, i, t, times, anticipation, horizons, m, dynamic)
                    if z == 0.0:
                        continue
                    mass[c, m] += w[i, t]
                    rhs[c, m] += w[i, t] * u1[t]
                    for a in range(P):
                        C[c, m, a] += w[i, t] * (X[t, a] - xm[a])


@njit(parallel=True, cache=True)
def cluster_scores(
    g, y, x, w, columns, theta, times, anticipation, horizons, dynamic, beta, H, mass, out, bad
):
    """Each unit's second-stage score, corrected for the first stage and scaled
    by the indicators' mass: ``out[unit, m]``."""
    n, T = y.shape
    P = columns.shape[0]
    M = beta.shape[0]
    T1 = T - 1
    chunks = (n + CHUNK - 1) // CHUNK
    for c in prange(chunks):
        X = np.empty((T, P))
        xm = np.zeros(P)
        u1 = np.zeros(T)
        s1 = np.zeros(P)
        obs = np.zeros(T, dtype=np.bool_)
        z = np.zeros(M)
        for i in range(c * CHUNK, min(n, (c + 1) * CHUNK)):
            if _first_stage(
                g, y, x, w, i, T1, columns, theta, times, anticipation, X, xm, u1, s1, obs
            ):
                bad[c] = 1
                continue
            score2 = np.zeros(M)
            for t in range(T):
                fitted = 0.0
                for m in range(M):
                    z[m] = _indicator(g, i, t, times, anticipation, horizons, m, dynamic)
                    if not obs[t]:
                        z[m] = 0.0
                    fitted += z[m] * beta[m]
                u2 = u1[t] - fitted
                for m in range(M):
                    score2[m] += w[i, t] * u2 * z[m]
            for m in range(M):
                adjust = 0.0
                for a in range(P):
                    adjust += s1[a] * H[a, m]
                out[i, m] = (score2[m] - adjust) / mass[m]


def _arrays(raw):
    g, y, x, w = raw
    g = np.ascontiguousarray(g, dtype=np.int64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    x = np.ascontiguousarray(x, dtype=np.float64)
    w = np.ascontiguousarray(w, dtype=np.float64)
    return g, y, x, w


def _chunks(n):
    return (n + CHUNK - 1) // CHUNK


class TwoStageKernels:
    """The compiled passes for a store: ``times`` are its period labels."""

    def __init__(self, times, anticipation):
        self.times = np.ascontiguousarray(times, dtype=np.int64)
        self.anticipation = int(anticipation)

    def reduce(self, batches, n_columns, clean):
        """(Q, q, norms, Qc, qc) over the batches of (g, y, x, w), all columns kept."""
        columns = np.arange(n_columns, dtype=np.int64)
        P = n_columns
        Q, norms, Qc, qc = np.zeros((P, P)), np.zeros(P), np.zeros((P, P)), np.zeros(P)
        q = np.zeros(P)
        for raw in batches():
            g, y, x, w = _arrays(raw)
            wc = w * clean(g) if clean is not None else w
            k = _chunks(len(g))
            parts = dict(
                Q=np.zeros((k, P, P)),
                q=np.zeros((k, P)),
                norms=np.zeros((k, P)),
                Qc=np.zeros((k, P, P)),
                qc=np.zeros((k, P)),
                bad=np.zeros(k, dtype=np.int8),
            )
            reduce_moments(
                y, x, w, wc, columns, clean is not None, parts["Q"], parts["q"], parts["norms"], parts["Qc"],
                parts["qc"], parts["bad"],
            )  # fmt: skip
            if parts["bad"].any():
                raise ValueError("each group requires a positive-weight estimation observation")
            Q += parts["Q"].sum(axis=0)
            q += parts["q"].sum(axis=0)
            norms += parts["norms"].sum(axis=0)
            Qc += parts["Qc"].sum(axis=0)
            qc += parts["qc"].sum(axis=0)
        return Q, q, norms, Qc, qc

    def second_stage(self, batches, keep, theta, horizons, dynamic):
        """(mass, rhs, C): the sums of the second stage's normal equations."""
        columns = np.ascontiguousarray(keep, dtype=np.int64)
        M, P = len(horizons), len(columns)
        mass, rhs, C = np.zeros(M), np.zeros(M), np.zeros((M, P))
        hz = np.ascontiguousarray(horizons, dtype=np.int64)
        for raw in batches():
            g, y, x, w = _arrays(raw)
            k = _chunks(len(g))
            parts = (
                np.zeros((k, M)),
                np.zeros((k, M)),
                np.zeros((k, M, P)),
                np.zeros(k, dtype=np.int8),
            )
            second_stage_sums(
                g, y, x, w, columns, theta, self.times, self.anticipation, hz, dynamic, *parts
            )
            if parts[3].any():
                raise ValueError("requires at least one untreated observation per unit")
            mass += parts[0].sum(axis=0)
            rhs += parts[1].sum(axis=0)
            C += parts[2].sum(axis=0)
        return mass, rhs, C

    def scores(self, raw, keep, theta, horizons, dynamic, beta, H, mass):
        columns = np.ascontiguousarray(keep, dtype=np.int64)
        g, y, x, w = _arrays(raw)
        out = np.zeros((len(g), len(beta)))
        bad = np.zeros(_chunks(len(g)), dtype=np.int8)
        cluster_scores(
            g, y, x, w, columns, theta, self.times, self.anticipation,
            np.ascontiguousarray(horizons, dtype=np.int64), dynamic, beta,
            np.ascontiguousarray(H), mass, out, bad,
        )  # fmt: skip
        if bad.any():
            raise ValueError("requires at least one untreated observation per unit")
        return out
