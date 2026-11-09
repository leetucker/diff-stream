"""Compiled passes of the extended TWFE regression.

The expanded design has a column for every cohort, period, cohort-by-covariate,
period-by-covariate and treatment-cell effect, but a row has only a handful of
nonzero entries: its cohort and period, the covariates interacted with each,
and the covariates (and a one) in its treatment cell. The NumPy code builds the
dense design for a batch of units and multiplies it; the kernels here visit
only the nonzero entries, and remove the unit means (``fixed_effects="unit"``)
analytically.

Units are processed in fixed-size chunks whose partial sums are added in order,
so results do not depend on the number of threads. The NumPy code in
``etwfe.py`` remains the reference; the tests compare the two.
"""

import numpy as np
from numba import njit, prange

ENABLED = True  # tests switch it off to compare with the array path
CHUNK = 64  # units per chunk
PARTIAL_BYTES = 128e6  # memory of the chunks' partial Gram matrices


@njit(cache=True, inline="always")
def _row(x, i, t, gi, K, G, T, cohort_fe, base_c, base_t, offset, cell_at, cols, vals):
    """The nonzero entries of one row of the design: fills ``cols`` and ``vals``
    and returns their number and the row's treatment cell (-1 for none)."""
    n = 0
    at = 0
    if cohort_fe:
        cols[n] = gi
        vals[n] = 1.0
        n += 1
        at = G
    if t >= 1:
        cols[n] = at + t - 1
        vals[n] = 1.0
        n += 1
    cell = cell_at[gi, t]
    for k in range(K):
        v = x[i, t, k]
        if not np.isfinite(v):
            v = 0.0
        cols[n] = base_c + gi * K + k
        vals[n] = v
        n += 1
        if t >= 1:
            cols[n] = base_t + (t - 1) * K + k
            vals[n] = v
            n += 1
    if cell >= 0:
        cols[n] = offset + cell * (K + 1)
        vals[n] = 1.0
        n += 1
        for k in range(K):
            v = x[i, t, k]
            cols[n] = offset + cell * (K + 1) + 1 + k
            vals[n] = v if np.isfinite(v) else 0.0
            n += 1
    return n, cell


@njit(parallel=True, cache=True)
def etwfe_moments(
    gidx, y, x, w, G, cohort_fe, unit_fe, base_c, base_t, offset, cell_at, chunk, Q, q, norms,
    means, cdiag, bad,
):  # fmt: skip
    """Gram matrix and moments of the regression, the unit-centered column
    norms, and the weighted covariate sums of each treatment cell."""
    n, T = y.shape
    K = x.shape[2]
    P = Q.shape[1]
    chunks = Q.shape[0]
    nz = 3 * K + 3
    for c in prange(chunks):
        cols = np.empty((T, nz), dtype=np.int64)
        vals = np.empty((T, nz))
        count = np.zeros(T, dtype=np.int64)
        s = np.zeros(P)
        wn = np.zeros(P)
        seen = np.zeros(P, dtype=np.bool_)
        touched = np.empty(T * nz, dtype=np.int64)
        for i in range(c * chunk, min(n, (c + 1) * chunk)):
            mass = 0.0
            ysum = 0.0
            ntouch = 0
            gi = gidx[i]
            for t in range(T):
                m, cell = _row(
                    x, i, t, gi, K, G, T, cohort_fe, base_c, base_t, offset, cell_at,
                    cols[t], vals[t],
                )  # fmt: skip
                count[t] = m
                wt = w[i, t]
                yt = y[i, t] if np.isfinite(y[i, t]) else 0.0
                mass += wt
                ysum += wt * yt
                for a in range(m):
                    ca = cols[t, a]
                    wa = wt * vals[t, a]
                    q[c, ca] += wa * yt
                    if unit_fe:
                        norms[c, ca] += wa * vals[t, a]
                        s[ca] += wa
                        wn[ca] += wt
                        if not seen[ca]:
                            seen[ca] = True
                            touched[ntouch] = ca
                            ntouch += 1
                    for b in range(m):
                        Q[c, ca, cols[t, b]] += wa * vals[t, b]
                if cell >= 0:
                    means[c, cell, 0] += wt
                    for k in range(K):
                        v = x[i, t, k]
                        means[c, cell, 1 + k] += wt * (v if np.isfinite(v) else 0.0)
            if unit_fe:
                if mass <= 0.0:
                    bad[c] = 1
                else:
                    for a in range(ntouch):
                        ca = touched[a]
                        q[c, ca] -= s[ca] * ysum / mass
                        for b in range(ntouch):
                            cb = touched[b]
                            Q[c, ca, cb] -= s[ca] * s[cb] / mass
                    # the centered sum of squares itself, without the cancellation
                    # of the subtraction above: a column the unit means remove
                    # entirely must come out as zero
                    for t in range(T):
                        for a in range(count[t]):
                            ca = cols[t, a]
                            d = vals[t, a] - s[ca] / mass
                            cdiag[c, ca] += w[i, t] * d * d
                    for a in range(ntouch):
                        ca = touched[a]
                        dbar = s[ca] / mass
                        cdiag[c, ca] += dbar * dbar * (mass - wn[ca])
                for a in range(ntouch):
                    s[touched[a]] = 0.0
                    wn[touched[a]] = 0.0
                    seen[touched[a]] = False


@njit(parallel=True, cache=True)
def etwfe_scores(
    gidx, y, x, w, G, cohort_fe, unit_fe, base_c, base_t, offset, cell_at, coef, H, out, bad
):  # fmt: skip
    """Each unit's score of the regression times H: ``out[unit, :]``."""
    n, T = y.shape
    K = x.shape[2]
    M = H.shape[1]
    nz = 3 * K + 3
    chunks = (n + CHUNK - 1) // CHUNK
    for c in prange(chunks):
        cols = np.empty((T, nz), dtype=np.int64)
        vals = np.empty((T, nz))
        count = np.zeros(T, dtype=np.int64)
        resid = np.zeros(T)
        for i in range(c * CHUNK, min(n, (c + 1) * CHUNK)):
            gi = gidx[i]
            mass = 0.0
            rsum = 0.0
            for t in range(T):
                m, cell = _row(
                    x, i, t, gi, K, G, T, cohort_fe, base_c, base_t, offset, cell_at,
                    cols[t], vals[t],
                )  # fmt: skip
                count[t] = m
                fit = 0.0
                for a in range(m):
                    fit += vals[t, a] * coef[cols[t, a]]
                yt = y[i, t] if np.isfinite(y[i, t]) else 0.0
                resid[t] = yt - fit
                mass += w[i, t]
                rsum += w[i, t] * resid[t]
            if unit_fe:
                if mass <= 0.0:
                    bad[c] = 1
                    continue
                rsum /= mass
            else:
                rsum = 0.0
            for t in range(T):
                we = w[i, t] * (resid[t] - rsum)
                if we == 0.0:
                    continue
                for a in range(count[t]):
                    wa = we * vals[t, a]
                    ca = cols[t, a]
                    for m in range(M):
                        out[i, m] += wa * H[ca, m]


def _arrays(raw):
    g, y, x, w = raw
    return (
        np.ascontiguousarray(y, dtype=np.float64),
        np.ascontiguousarray(x, dtype=np.float64),
        np.ascontiguousarray(w, dtype=np.float64),
    )


class ETWFEKernels:
    """The compiled passes for an ``etwfe.ExtendedDesign``."""

    def __init__(self, model):
        self.model = model
        self.P = len(model.names)
        self.G = len(model.groups)
        self.cohort_fe = model.fixed_effects == "cohort"
        T = model.store.periods
        K = model.K
        self.base_c = (self.G if self.cohort_fe else 0) + T - 1
        self.base_t = self.base_c + self.G * K
        self.offset = model.offset
        self.cell_at = np.ascontiguousarray(model.cell_at)
        # chunks per batch: the partial Gram matrices must fit
        self.max_chunks = int(PARTIAL_BYTES // (8 * self.P * self.P))

    @property
    def usable(self):
        return ENABLED and self.max_chunks >= 2

    def _chunking(self, n):
        chunks = min((n + CHUNK - 1) // CHUNK, self.max_chunks)
        return chunks, (n + chunks - 1) // chunks

    def group_index(self, g):
        return np.searchsorted(self.model.group_array, g).astype(np.int64)

    def moments(self, batches):
        """(Q, q, norms, means): the Gram matrix, moments, column norms and
        covariate sums of each cell."""
        model = self.model
        P, M, K = self.P, len(model.cells), model.K
        Q, q, norms = np.zeros((P, P)), np.zeros(P), np.zeros(P)
        means = np.zeros((M, K + 1))
        cdiag = np.zeros(P)
        for raw in batches():
            y, x, w = _arrays(raw)
            g = raw[0]
            chunks, size = self._chunking(len(g))
            parts = (
                np.zeros((chunks, P, P)),
                np.zeros((chunks, P)),
                np.zeros((chunks, P)),
                np.zeros((chunks, M, K + 1)),
                np.zeros((chunks, P)),
                np.zeros(chunks, dtype=np.int8),
            )
            etwfe_moments(
                self.group_index(g), y, x, w, self.G,
                self.cohort_fe, model.fixed_effects == "unit", self.base_c, self.base_t,
                self.offset, self.cell_at, size, *parts,
            )  # fmt: skip
            if parts[5].any():
                raise ValueError("each group requires a positive-weight estimation observation")
            Q += parts[0].sum(axis=0)
            q += parts[1].sum(axis=0)
            norms += parts[2].sum(axis=0)
            means += parts[3].sum(axis=0)
            cdiag += parts[4].sum(axis=0)
        if model.fixed_effects == "unit":
            # columns the unit means remove entirely are exactly zero, as in
            # the array path; the subtraction leaves rounding noise
            Q[np.diag_indices(P)] = cdiag
            dead = cdiag <= norms * 1e-24
            Q[dead, :] = 0.0
            Q[:, dead] = 0.0
            q[dead] = 0.0
        return Q, q, norms, means

    def scores(self, raw, coef, H_full):
        """The batch's scores times H (units by cells); ``H_full`` has a row for
        every column of the design (zero for the dropped ones)."""
        model = self.model
        y, x, w = _arrays(raw)
        out = np.zeros((len(y), H_full.shape[1]))
        bad = np.zeros((len(y) + CHUNK - 1) // CHUNK, dtype=np.int8)
        etwfe_scores(
            self.group_index(raw[0]), y, x, w, self.G, self.cohort_fe,
            model.fixed_effects == "unit", self.base_c, self.base_t, self.offset, self.cell_at,
            np.ascontiguousarray(coef), np.ascontiguousarray(H_full), out, bad,
        )  # fmt: skip
        if bad.any():
            raise ValueError("each group requires a positive-weight estimation observation")
        return out
