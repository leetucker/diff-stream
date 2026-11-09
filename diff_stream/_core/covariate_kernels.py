"""Compiled passes of Sun-Abraham and BJS with covariates or extra fixed effects.

Both fit a within-unit regression on a design of period dummies and covariates
(``linear.RowDesign``); the NumPy code builds that design for each batch and
loops over cohorts and aggregation groups in Python. The kernels read the raw
batch arrays, build one unit's design rows as they go, and do the same sums in
one compiled call per batch. Units are processed in fixed-size chunks whose
partial sums are added in order, so results do not depend on the number of
threads. The NumPy code in ``linear.py`` and ``unbalanced.py`` remains the
reference; the tests compare the two.
"""

import numpy as np
from numba import njit, prange

from .twostage_kernels import CHUNK, _design, _first_stage

ENABLED = True  # tests switch it off to compare with the array path


def _chunks(n):
    return (n + CHUNK - 1) // CHUNK


def _arrays(raw):
    g, y, x, w = raw
    return (
        np.ascontiguousarray(y, dtype=np.float64),
        np.ascontiguousarray(x, dtype=np.float64),
        np.ascontiguousarray(w, dtype=np.float64),
    )


# ------------------------------------------------------------------ Sun-Abraham


@njit(cache=True, inline="always")
def _sunab_row(x, i, t, T1, columns, cell, K, X, cols, vals):
    """Row (i, t) of the design: the kept columns, then the cohort-period cell."""
    _design(x, i, t, T1, columns, X)
    for a in range(K):
        cols[a] = a
        vals[a] = X[t, a]
    n = K
    if cell >= 0:
        cols[n] = K + cell
        vals[n] = 1.0
        n += 1
    return n


@njit(parallel=True, cache=True)
def sunab_moments(gidx, y, x, w, columns, cell_at, chunk, Q, q, bad):
    """Within-unit Gram matrix and moments of [kept columns | cell indicators]."""
    n, T = y.shape
    K = columns.shape[0]
    P = Q.shape[1]
    T1 = T - 1
    for c in prange(Q.shape[0]):
        X = np.empty((T, K))
        cols = np.empty((T, K + 1), dtype=np.int64)
        vals = np.empty((T, K + 1))
        count = np.zeros(T, dtype=np.int64)
        s = np.zeros(P)
        seen = np.zeros(P, dtype=np.bool_)
        touched = np.empty(P, dtype=np.int64)
        for i in range(c * chunk, min(n, (c + 1) * chunk)):
            mass = 0.0
            ysum = 0.0
            ntouch = 0
            for a in range(K):
                touched[ntouch] = a
                ntouch += 1
                seen[a] = True
            gi = gidx[i]
            for t in range(T):
                count[t] = _sunab_row(x, i, t, T1, columns, cell_at[gi, t], K, X, cols[t], vals[t])
                wt = w[i, t]
                yt = y[i, t] if np.isfinite(y[i, t]) else 0.0
                mass += wt
                ysum += wt * yt
                for a in range(count[t]):
                    ca = cols[t, a]
                    wa = wt * vals[t, a]
                    q[c, ca] += wa * yt
                    s[ca] += wa
                    if not seen[ca]:
                        seen[ca] = True
                        touched[ntouch] = ca
                        ntouch += 1
                    for b in range(count[t]):
                        Q[c, ca, cols[t, b]] += wa * vals[t, b]
            if mass <= 0.0:
                bad[c] = 1
            else:
                for a in range(ntouch):
                    ca = touched[a]
                    q[c, ca] -= s[ca] * ysum / mass
                    for b in range(ntouch):
                        cb = touched[b]
                        Q[c, ca, cb] -= s[ca] * s[cb] / mass
            for a in range(ntouch):
                s[touched[a]] = 0.0
                seen[touched[a]] = False


@njit(parallel=True, cache=True)
def sunab_scores(gidx, y, x, w, columns, cell_at, coef, H, out, bad):
    """Each unit's score times H: ``out[unit, :]``."""
    n, T = y.shape
    K = columns.shape[0]
    M = H.shape[1]
    T1 = T - 1
    for c in prange((n + CHUNK - 1) // CHUNK):
        X = np.empty((T, K))
        cols = np.empty((T, K + 1), dtype=np.int64)
        vals = np.empty((T, K + 1))
        count = np.zeros(T, dtype=np.int64)
        resid = np.zeros(T)
        for i in range(c * CHUNK, min(n, (c + 1) * CHUNK)):
            gi = gidx[i]
            mass = 0.0
            rsum = 0.0
            for t in range(T):
                count[t] = _sunab_row(x, i, t, T1, columns, cell_at[gi, t], K, X, cols[t], vals[t])
                fit = 0.0
                for a in range(count[t]):
                    fit += vals[t, a] * coef[cols[t, a]]
                yt = y[i, t] if np.isfinite(y[i, t]) else 0.0
                resid[t] = yt - fit
                mass += w[i, t]
                rsum += w[i, t] * resid[t]
            if mass <= 0.0:
                bad[c] = 1
                continue
            rsum /= mass
            for t in range(T):
                we = w[i, t] * (resid[t] - rsum)
                if we == 0.0:
                    continue
                for a in range(count[t]):
                    wa = we * vals[t, a]
                    ca = cols[t, a]
                    for m in range(M):
                        out[i, m] += wa * H[ca, m]


class SunAbrahamKernels:
    def __init__(self, store, cells, keep):
        self.store, self.cells = store, cells
        self.columns = np.ascontiguousarray(keep, dtype=np.int64)
        self.K, self.M = len(self.columns), len(cells)
        self.groups = np.asarray(sorted(store.gc))
        t0 = np.asarray(store.times)
        self.cell_at = np.full((len(self.groups), store.periods), -1, dtype=np.int64)
        for j, (cg, t) in enumerate(cells):
            self.cell_at[np.searchsorted(self.groups, cg), int(np.searchsorted(t0, t))] = j

    def _index(self, g):
        return np.searchsorted(self.groups, g).astype(np.int64)

    def moments(self, batches):
        P = self.K + self.M
        Q, q = np.zeros((P, P)), np.zeros(P)
        for raw in batches():
            y, x, w = _arrays(raw)
            k = _chunks(len(y))
            parts = (np.zeros((k, P, P)), np.zeros((k, P)), np.zeros(k, dtype=np.int8))
            sunab_moments(self._index(raw[0]), y, x, w, self.columns, self.cell_at, CHUNK, *parts)
            if parts[2].any():
                raise ValueError("each group requires a positive-weight estimation observation")
            Q += parts[0].sum(axis=0)
            q += parts[1].sum(axis=0)
        return Q, q

    def scores(self, raw, coef, H):
        y, x, w = _arrays(raw)
        out = np.zeros((len(y), H.shape[1]))
        bad = np.zeros(_chunks(len(y)), dtype=np.int8)
        sunab_scores(
            self._index(raw[0]), y, x, w, self.columns, self.cell_at,
            np.ascontiguousarray(coef), np.ascontiguousarray(H), out, bad,
        )  # fmt: skip
        if bad.any():
            raise ValueError("each group requires a positive-weight estimation observation")
        return out


# ------------------------------------------------------------------------- BJS


@njit(parallel=True, cache=True)
def bjs_cell_sums(
    g, gidx, y, x, w, columns, theta, times, anticipation, cell_at, counts, beta, D, bad
):  # fmt: skip
    """Cell effects and their derivative with respect to the first stage."""
    n, T = y.shape
    P = columns.shape[0]
    T1 = T - 1
    for c in prange(beta.shape[0]):
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
            gi = gidx[i]
            for t in range(T):
                j = cell_at[gi, t]
                if j < 0 or not obs[t]:
                    continue
                sw = w[i, t] / counts[j]
                beta[c, j] += sw * u1[t]
                for a in range(P):
                    D[c, j, a] += sw * (X[t, a] - xm[a])


@njit(parallel=True, cache=True)
def bjs_totals(
    g, gidx, y, x, w, columns, theta, times, anticipation, cell_at, cell_slot, slot_label,
    slot_ptr, Aw, denom, numer, bad,
):  # fmt: skip
    """Sums of squares and cross products of each unit's weighted totals and
    effects within each residual-averaging group (``ObservedBJSInference.covariance``)."""
    n, T = y.shape
    P = columns.shape[0]
    Pc = Aw.shape[0]
    T1 = T - 1
    for c in prange(denom.shape[0]):
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
            gi = gidx[i]
            lo = slot_ptr[gi]
            S = slot_ptr[gi + 1] - lo
            tot = np.zeros((S, Pc))
            eff = np.zeros((S, Pc))
            for t in range(T):
                j = cell_at[gi, t]
                if j < 0:
                    continue
                sl = cell_slot[j]
                if obs[t]:
                    for a in range(Pc):
                        tot[sl, a] += w[i, t] * Aw[a, j]
                for a in range(Pc):
                    eff[sl, a] += w[i, t] * u1[t] * Aw[a, j]
            for sl in range(S):
                r = slot_label[lo + sl]
                for a in range(Pc):
                    denom[c, a, r] += tot[sl, a] * tot[sl, a]
                    numer[c, a, r] += tot[sl, a] * eff[sl, a]


@njit(parallel=True, cache=True)
def bjs_scores(
    g, gidx, y, x, w, columns, theta, times, anticipation, cell_at, cell_slot, slot_label,
    slot_ptr, Aw, derivative, averages, denom, leaveout, out, invalid, bad,
):  # fmt: skip
    """Each unit's score of the aggregates: ``out[unit, :]``; ``invalid[chunk, a]``
    marks aggregates with a leave-out residual group of one contributor."""
    n, T = y.shape
    P = columns.shape[0]
    Pc = Aw.shape[0]
    T1 = T - 1
    for c in prange((n + CHUNK - 1) // CHUNK):
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
            gi = gidx[i]
            lo = slot_ptr[gi]
            S = slot_ptr[gi + 1] - lo
            tot = np.zeros((S, Pc))
            eff = np.zeros((S, Pc))
            for t in range(T):
                j = cell_at[gi, t]
                if j < 0:
                    continue
                sl = cell_slot[j]
                if obs[t]:
                    for a in range(Pc):
                        tot[sl, a] += w[i, t] * Aw[a, j]
                for a in range(Pc):
                    eff[sl, a] += w[i, t] * u1[t] * Aw[a, j]
            for a in range(Pc):
                acc = 0.0
                for p in range(P):
                    acc += s1[p] * derivative[a, p]
                out[i, a] = -acc
            for sl in range(S):
                r = slot_label[lo + sl]
                for a in range(Pc):
                    contribution = eff[sl, a] - tot[sl, a] * averages[a, r]
                    if leaveout:
                        lev = 0.0
                        if denom[a, r] > 0.0:
                            lev = tot[sl, a] * tot[sl, a] / denom[a, r]
                        if abs(tot[sl, a]) > 0.0 and lev >= 1.0 - 1e-12:
                            invalid[c, a] = 1
                        if lev < 1.0 - 1e-12:
                            contribution = contribution / (1.0 - lev)
                        else:
                            contribution = 0.0
                    out[i, a] += contribution


class BJSKernels:
    """Compiled passes of BJS with a ``RowDesign`` (``linear.fit_bjs`` and the
    aggregation covariance in ``unbalanced.ObservedBJSInference``)."""

    def __init__(self, store, cells, keep, theta, anticipation):
        self.store, self.cells = store, cells
        self.columns = np.ascontiguousarray(keep, dtype=np.int64)
        self.theta = np.ascontiguousarray(theta, dtype=np.float64)
        self.anticipation = int(anticipation)
        self.times = np.ascontiguousarray(store.times, dtype=np.int64)
        self.groups = np.asarray(sorted(store.gc))
        self.cell_at = np.full((len(self.groups), store.periods), -1, dtype=np.int64)
        for j, (cg, t) in enumerate(cells):
            self.cell_at[np.searchsorted(self.groups, cg), int(np.searchsorted(self.times, t))] = j

    def _arrays(self, raw):
        y, x, w = _arrays(raw)
        g = np.ascontiguousarray(raw[0], dtype=np.int64)
        return g, np.searchsorted(self.groups, g).astype(np.int64), y, x, w

    def cell_sums(self, batches, counts):
        P, M = len(self.columns), len(self.cells)
        beta, D = np.zeros(M), np.zeros((M, P))
        counts = np.ascontiguousarray(counts, dtype=np.float64)
        for raw in batches():
            g, gi, y, x, w = self._arrays(raw)
            k = _chunks(len(g))
            parts = (np.zeros((k, M)), np.zeros((k, M, P)), np.zeros(k, dtype=np.int8))
            bjs_cell_sums(
                g, gi, y, x, w, self.columns, self.theta, self.times, self.anticipation,
                self.cell_at, counts, *parts,
            )  # fmt: skip
            if parts[2].any():
                raise ValueError("BJS requires at least one untreated observation per unit")
            beta += parts[0].sum(axis=0)
            D += parts[1].sum(axis=0)
        return beta, D

    def grouping(self, labels, counts, A):
        """Per-cohort slots for the residual-averaging groups of ``labels`` (cell -> group)."""
        G = len(self.groups)
        slot_ptr = np.zeros(G + 1, dtype=np.int64)
        cell_slot = np.zeros(len(self.cells), dtype=np.int64)
        slot_label = []
        for gi, cg in enumerate(self.groups):
            seen = {}
            for j, (cell_cohort, _) in enumerate(self.cells):
                if cell_cohort == cg:
                    r = labels[j]
                    cell_slot[j] = seen.setdefault(r, len(seen))
            slot_label.extend(seen)
            slot_ptr[gi + 1] = len(slot_label)
        return (
            cell_slot,
            np.asarray(slot_label, dtype=np.int64),
            slot_ptr,
            np.ascontiguousarray(A / np.asarray(counts)[None, :]),
        )

    def totals(self, batches, grouping, R):
        cell_slot, slot_label, slot_ptr, Aw = grouping
        Pc = Aw.shape[0]
        denom, numer = np.zeros((Pc, R)), np.zeros((Pc, R))
        for raw in batches():
            g, gi, y, x, w = self._arrays(raw)
            k = _chunks(len(g))
            parts = (np.zeros((k, Pc, R)), np.zeros((k, Pc, R)), np.zeros(k, dtype=np.int8))
            bjs_totals(
                g, gi, y, x, w, self.columns, self.theta, self.times, self.anticipation,
                self.cell_at, cell_slot, slot_label, slot_ptr, Aw, *parts,
            )  # fmt: skip
            if parts[2].any():
                raise ValueError("BJS requires at least one untreated observation per unit")
            denom += parts[0].sum(axis=0)
            numer += parts[1].sum(axis=0)
        return denom, numer

    def scores(self, raw, grouping, derivative, averages, denom, leaveout):
        cell_slot, slot_label, slot_ptr, Aw = grouping
        g, gi, y, x, w = self._arrays(raw)
        out = np.zeros((len(g), Aw.shape[0]))
        k = _chunks(len(g))
        invalid = np.zeros((k, Aw.shape[0]), dtype=np.int8)
        bad = np.zeros(k, dtype=np.int8)
        bjs_scores(
            g, gi, y, x, w, self.columns, self.theta, self.times, self.anticipation,
            self.cell_at, cell_slot, slot_label, slot_ptr, Aw,
            np.ascontiguousarray(derivative), np.ascontiguousarray(averages),
            np.ascontiguousarray(denom), bool(leaveout), out, invalid, bad,
        )  # fmt: skip
        if bad.any():
            raise ValueError("BJS requires at least one untreated observation per unit")
        return out, invalid.any(axis=0)
