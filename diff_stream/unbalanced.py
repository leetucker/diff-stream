"""Observed-row Sun–Abraham and BJS systems for unbalanced panels."""

import shutil
from tempfile import TemporaryDirectory

import numpy as np

from ._core.nuisance import solve_checked


def fit_sunab(store, cells, anticipation=0):
    """Exact saturated SA regression via independent cohort calendar effects.

    Unit intercepts absorb each cohort's calendar-effect normalization. Since
    every other observed cohort/date has an interaction, the regression splits
    into one unit/time FE fit per cohort. Subtract the never-treated calendar
    change from each treated cohort's change relative to g-1. This identity
    holds with arbitrary observed-row patterns; no balanced change shortcut or
    all-row interaction design is used.
    """
    T, M = store.periods, len(cells)
    systems = {}
    for g, counts in store.cell_counts.items():
        reference = g - anticipation - 1 - store.times[0] if g else 0
        if counts[reference] == 0 or (g == 0 and np.any(counts == 0)):
            raise ValueError(
                "Sun–Abraham interactions are unidentified: missing cohort reference or never-treated date"
            )
        free = np.flatnonzero((counts > 0) & (np.arange(T) != reference))
        Z = np.eye(T)[:, free]
        systems[g] = [Z, np.zeros((len(free), len(free))), np.zeros(len(free))]
    for g, y, w in store.batches(include_weights=True):
        for cg in np.unique(g):
            Z, Q, q = systems[cg]
            values = y[g == cg]
            sw = w[g == cg]
            obs = np.isfinite(values)
            n = obs.sum(axis=1)
            zbar = obs @ Z / n[:, None]
            raw = np.where(obs, values, 0)
            centered = np.where(obs, raw - raw.sum(axis=1)[:, None] / n[:, None], 0)
            Q += Z.T @ ((sw @ obs)[:, None] * Z) - zbar.T @ ((sw * n)[:, None] * zbar)
            q += Z.T @ (sw @ centered)
    coefficients, derivatives = {}, {}
    for g, (Z, Q, q) in systems.items():
        context = f"Sun–Abraham interactions are unidentified in cohort {g} (disconnected observed design)"
        b = solve_checked(Q, q, context) if len(q) else np.zeros(0)
        coefficients[g] = Z @ b
        ix = [j for j, (cg, t) in enumerate(cells) if cg == g] if g else list(range(M))
        D = np.zeros((len(ix), Z.shape[1]))
        for k, j in enumerate(ix):
            cg, t = cells[j]
            ti, bi = t - store.times[0], cg - anticipation - 1 - store.times[0]
            D[k] = (Z[ti] - Z[bi]) * (1 if g else -1)
        H = solve_checked(Q, D.T, context) if len(q) else np.zeros((0, len(ix)))
        derivatives[g] = (ix, H)
    beta = np.array(
        [
            coefficients[g][t - store.times[0]]
            - coefficients[g][g - anticipation - 1 - store.times[0]]
            - coefficients[0][t - store.times[0]]
            + coefficients[0][g - anticipation - 1 - store.times[0]]
            for g, t in cells
        ]
    )
    counts = np.array([store.cell_weights[g][t - store.times[0]] for g, t in cells], float)
    V = np.zeros((M, M))
    for g, y, w in store.batches(include_weights=True):
        scores = np.zeros((len(g), M))
        for cg in np.unique(g):
            mask = g == cg
            obs = np.isfinite(y[mask])
            raw = np.where(obs, y[mask] - coefficients[cg], 0)
            residual = np.where(obs, raw - raw.sum(axis=1)[:, None] / obs.sum(axis=1)[:, None], 0)
            ix, H = derivatives[cg]
            scores[np.ix_(np.flatnonzero(mask), ix)] = w[mask, None] * (
                residual @ systems[cg][0] @ H
            )
        V += scores.T @ scores
    return beta, counts, V


def residuals(g, y, times, Z, theta, anticipation=0):
    observed = np.isfinite(y)
    untreated = observed & ((g[:, None] == 0) | (times[None, :] < g[:, None] - anticipation))
    n0 = untreated.sum(axis=1)
    if np.any(n0 == 0):
        raise ValueError(
            "BJS requires at least one untreated observation per unit; exclude units without one explicitly"
        )
    zbar = untreated @ Z / n0[:, None]
    raw = np.where(observed, y - Z @ theta, 0)
    alpha = (raw * untreated).sum(axis=1) / n0
    r = np.where(observed, raw - alpha[:, None], 0)
    return observed, untreated, zbar, r


def fit_bjs(store, cells, avgeffectsby, leaveout, workdir, anticipation=0):
    T, M = store.periods, len(cells)
    Z = np.eye(T)[:, 1:]
    Q, q = np.zeros((T - 1, T - 1)), np.zeros(T - 1)
    for g, y, w in store.batches(include_weights=True):
        _, use, zm, centered = residuals(g, y, store.times, Z, np.zeros(T - 1), anticipation)
        n0 = use.sum(axis=1)
        Q += Z.T @ ((w @ use)[:, None] * Z) - zm.T @ ((w * n0)[:, None] * zm)
        q += Z.T @ (w @ (centered * use))
    theta = solve_checked(Q, q, "BJS untreated time effects (disconnected observed design)")
    counts = np.array([store.cell_weights[g][t - store.times[0]] for g, t in cells], float)
    beta, D = np.zeros(M), np.zeros((M, T - 1))
    for g, y, w in store.batches(include_weights=True):
        observed, _, zm, r = residuals(g, y, store.times, Z, theta, anticipation)
        for j, (cg, t) in enumerate(cells):
            ti = t - store.times[0]
            mask = (g == cg) & observed[:, ti]
            beta[j] += w[mask] @ r[mask, ti] / counts[j]
            D[j] += w[mask] @ (Z[ti] - zm[mask]) / counts[j]
    H = solve_checked(Q, D.T, "BJS untreated time effects").T
    context = ObservedBJSInference(
        store, cells, counts, Z, theta, H, avgeffectsby, leaveout, workdir, anticipation
    )
    try:
        V = context.covariance(np.eye(M), allow_unidentified=avgeffectsby != "cohort_time")
    except BaseException:
        context.cleanup()
        raise
    return beta, counts, V, context


class ObservedBJSInference:
    """Disk-backed, bounded cluster inference including missingness patterns.

    The result owns these sorted observed-row partitions. Each aggregation uses
    two streaming passes; no unit-by-effect score matrix is saved or retained.
    """

    def __init__(
        self,
        store,
        cells,
        counts,
        Z,
        theta,
        H,
        avgeffectsby,
        leaveout,
        workdir,
        anticipation=0,
        *,
        model=None,
    ):
        self.workspace = TemporaryDirectory(prefix="diff-stream-bjs-", dir=workdir)
        self.store, self.cells, self.counts = store, cells, counts
        self.Z, self.theta, self.H = Z, theta, H
        self.model = model
        self.avgeffectsby, self.leaveout = avgeffectsby, leaveout
        self.anticipation = anticipation
        self.closed = False
        paths = []
        for path in store.paths:
            paths.append(shutil.move(str(path), self.workspace.name + "/" + path.name))
        store.paths = paths

    def cleanup(self):
        self.workspace.cleanup()
        self.closed = True

    def covariance(self, contrasts, *, allow_unidentified=False):
        if self.closed:
            raise ValueError("BJS aggregation data were removed by cleanup(); refit to aggregate")
        A = np.atleast_2d(contrasts)
        P, M = A.shape
        self.store._guard(
            8
            * (
                8 * P * P
                + 8 * min(self.store.batch_units, self.store.n_units) * (P + self.store.periods)
            )
            + 32e6
            + (
                8
                * (
                    8
                    * min(self.store.batch_units, self.store.n_units)
                    * self.store.periods
                    * (len(self.model.keep) + 1)
                    + 8 * P * len(self.model.keep)
                )
                if self.model is not None
                else 0
            )
        )
        labels, groups = {}, {}
        for j, (g, t) in enumerate(self.cells):
            label = {"cohort_time": (g, t), "event_time": t - g, "cohort": g, "all": 0}[
                self.avgeffectsby
            ]
            r = labels.setdefault(label, len(labels))
            groups.setdefault((g, r), []).append(j)
        R = len(labels)
        denom, numer = np.zeros((P, R)), np.zeros((P, R))

        def batches():
            if self.model is not None:
                yield from self.model.batches()
            else:
                for g, y, sw in self.store.batches(include_weights=True):
                    yield g, y, None, sw

        def parts(g, y, X, sw):
            if self.model is not None:
                observed, untreated, _, resid, first_score = self.model.residuals(
                    g, y, X, sw, self.theta, self.anticipation
                )
            else:
                observed, untreated, _, resid = residuals(
                    g, y, self.store.times, self.Z, self.theta, self.anticipation
                )
                first_score = sw[:, None] * ((resid * untreated) @ self.Z)
                sw = np.broadcast_to(sw[:, None], y.shape)
            yield first_score
            for (cg, r), js in groups.items():
                mask = g == cg
                if not mask.any():
                    continue
                ti = np.array([self.cells[j][1] - self.store.times[0] for j in js])
                w = A[:, js] / self.counts[js]
                totals = (sw[mask][:, ti] * observed[mask][:, ti]) @ w.T
                effects = (sw[mask][:, ti] * resid[mask][:, ti]) @ w.T
                yield mask, r, totals, effects

        for g, y, X, sw in batches():
            iterator = parts(g, y, X, sw)
            next(iterator)
            for _mask, r, totals, effects in iterator:
                denom[:, r] += (totals**2).sum(axis=0)
                numer[:, r] += (totals * effects).sum(axis=0)
        averages = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)
        V, unidentified = np.zeros((P, P)), np.zeros(P, bool)
        derivative = A @ self.H
        for g, y, X, sw in batches():
            iterator = parts(g, y, X, sw)
            scores = -next(iterator) @ derivative.T
            for mask, r, totals, effects in iterator:
                contribution = effects - totals * averages[:, r]
                if self.leaveout:
                    lev = np.divide(
                        totals**2, denom[:, r], out=np.zeros_like(totals), where=denom[:, r] > 0
                    )
                    invalid = (np.abs(totals) > 0) & (lev >= 1 - 1e-12)
                    unidentified |= invalid.any(axis=0)
                    if invalid.any() and not allow_unidentified:
                        raise ValueError(
                            "BJS leave_out requires more than one contributing unit in each residual group; use a coarser residual_averaging"
                        )
                    contribution = np.divide(
                        contribution,
                        1 - lev,
                        out=np.zeros_like(contribution),
                        where=lev < 1 - 1e-12,
                    )
                scores[mask] += contribution
            V += scores.T @ scores
        V[unidentified, :] = np.nan
        V[:, unidentified] = np.nan
        return (V + V.T) / 2
