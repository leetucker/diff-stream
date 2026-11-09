"""Compiled passes of the Callaway-Sant'Anna fit on observed rows.

Unbalanced panels, panels with sampling weights that vary over time, and
repeated cross sections are fitted from observed rows: a comparison takes each
unit's rows at its two dates (the base and the target) that have an outcome.
``cs_unbalanced.py`` does that in NumPy, one comparison at a time, and each
pass over the staged units repeats a few hundred small array operations per
batch. The kernels here do the same sums in one compiled call per batch, in
parallel over comparisons, as ``cs_kernels`` does for balanced panels: each
comparison is handled by one thread and adds its rows in the order they come
(a unit's base row before its target row), so results do not depend on the
number of threads. The NumPy code remains the reference, and the tests compare
the two.
"""

import numpy as np
from numba import njit, prange

from .cs_kernels import DR, IPW, METHOD_CODES, REG, _expit  # noqa: F401

ENABLED = True  # tests switch it off to compare with the array path
CAP = 1.0 - 1e-6


@njit(cache=True, inline="always")
def _weight(w2, i, t):
    return w2[i, t] if w2.shape[1] > 1 else w2[i, 0]


@njit(parallel=True, cache=True)
def row_initial(g, y, x, w2, members, cohort, dates, counts, wsums, scale):
    """Counts, weight sums and design scales of each logit system's rows."""
    n = g.shape[0]
    K = x.shape[2]
    for c in prange(members.shape[0]):
        cnt = np.zeros(2, dtype=np.int64)
        ws = np.zeros(4)
        sc = np.zeros(K)
        for i in range(n):
            if members[c, g[i]] == 0:
                continue
            treated = g[i] == cohort[c]
            for pi in range(2):
                t = dates[c, pi]
                if not np.isfinite(y[i, t]):
                    continue
                wi = _weight(w2, i, t)
                if treated:
                    cnt[0] += 1
                    ws[0] += wi
                    ws[2] += wi * wi
                else:
                    cnt[1] += 1
                    ws[1] += wi
                    ws[3] += wi * wi
                for a in range(K):
                    sc[a] += wi * x[i, t, a] * x[i, t, a]
        counts[c, :] = cnt
        wsums[c, :] = ws
        scale[c, :] = sc


@njit(parallel=True, cache=True)
def row_newton(g, y, x, w2, members, cohort, dates, beta, active, loss, grad, hess, margin):
    """Logit loss, gradient, information and smallest margin of the active systems."""
    n = g.shape[0]
    K = x.shape[2]
    for c in prange(members.shape[0]):
        if not active[c]:
            continue
        total = 0.0
        low = np.inf
        gd = np.zeros(K)
        hs = np.zeros((K, K))
        for i in range(n):
            if members[c, g[i]] == 0:
                continue
            d = 1.0 if g[i] == cohort[c] else 0.0
            for pi in range(2):
                t = dates[c, pi]
                if not np.isfinite(y[i, t]):
                    continue
                wi = _weight(w2, i, t)
                eta = 0.0
                for a in range(K):
                    eta += x[i, t, a] * beta[c, a]
                e = np.exp(-abs(eta))
                p = 1.0 / (1.0 + e) if eta >= 0.0 else e / (1.0 + e)
                total += wi * (max(eta, 0.0) + np.log1p(e) - d * eta)
                r = wi * (d - p)
                v = wi * p * (1.0 - p)
                for a in range(K):
                    xa = x[i, t, a]
                    gd[a] += xa * r
                    xv = xa * v
                    for b in range(K):
                        hs[a, b] += xv * x[i, t, b]
                m = (2.0 * d - 1.0) * eta
                if m < low:
                    low = m
        loss[c] = total
        grad[c, :] = gd
        hess[c, :, :] = hs
        margin[c] = low


@njit(parallel=True, cache=True)
def row_linear(g, y, x, w2, members, dates, gram, rhs):
    """Weighted least squares moments of each (cohorts, date) stratum."""
    n = g.shape[0]
    K = x.shape[2]
    for c in prange(members.shape[0]):
        t = dates[c]
        gr = np.zeros((K, K))
        rh = np.zeros(K)
        for i in range(n):
            if members[c, g[i]] == 0 or not np.isfinite(y[i, t]):
                continue
            wi = _weight(w2, i, t)
            for a in range(K):
                xa = x[i, t, a] * wi
                rh[a] += xa * y[i, t]
                for b in range(K):
                    gr[a, b] += xa * x[i, t, b]
        gram[c, :, :] = gr
        rhs[c, :] = rh


@njit(inline="always")
def _row_terms(method, treated, post, odds, yv, m0, m1, m2, m3, wt, vt):
    """The comparison's terms for one row: weights (before sampling weights) and values."""
    tp = 1.0 if (treated and post) else 0.0
    tm = 1.0 if (treated and not post) else 0.0
    cp = 1.0 if (not treated and post) else 0.0
    cm = 1.0 if (not treated and not post) else 0.0
    d = 1.0 if treated else 0.0
    if method == IPW:
        wt[0], wt[1], wt[2], wt[3] = tp, tm, cp * odds, cm * odds
        vt[0] = vt[1] = vt[2] = vt[3] = yv
        return 4
    if method == REG:
        wt[0], wt[1], wt[2] = tp, tm, d
        vt[0] = vt[1] = yv
        vt[2] = m1 - m0
        return 3
    r = yv - (m1 if post else m0)
    wt[0], wt[1], wt[2], wt[3] = tp, tm, cp * odds, cm * odds
    wt[4], wt[5], wt[6], wt[7] = d, tp, d, tm
    vt[0] = vt[1] = vt[2] = vt[3] = r
    vt[4] = vt[5] = m3 - m1
    vt[6] = vt[7] = m2 - m0
    return 8


@njit(parallel=True, cache=True)
def row_moments(
    g,
    y,
    x,
    w2,
    members,
    cohort,
    dates,
    method,
    trim,
    has_system,
    beta,
    models,
    weight,
    value,
    xw,
    xv,
    hess,
    strata,
    sw,
    sw2,
    trimmed,
    capped,
    cws,
    pmin,
    pmax,
):
    """The sums of one pass after the nuisance fits (``RowFunctional.update``)."""
    n = g.shape[0]
    K = x.shape[2]
    for c in prange(members.shape[0]):
        Wt = np.zeros(8)
        Vt = np.zeros(8)
        a_w = np.zeros(8)
        a_v = np.zeros(8)
        a_x = np.zeros((8, K))
        a_xv = np.zeros((8, K))
        a_h = np.zeros((K, K))
        a_n = np.zeros(4, dtype=np.int64)
        a_s = np.zeros(4)
        a_s2 = np.zeros(4)
        a_c = np.zeros(2)
        lo = np.inf
        hi = -np.inf
        ntrim = 0
        ncap = 0
        nt = 0
        for i in range(n):
            if members[c, g[i]] == 0:
                continue
            treated = g[i] == cohort[c]
            for pi in range(2):
                t = dates[c, pi]
                if not np.isfinite(y[i, t]):
                    continue
                post = pi == 1
                wi = _weight(w2, i, t)
                stratum = (2 if treated else 0) + (1 if post else 0)
                a_n[stratum] += 1
                a_s[stratum] += wi
                a_s2[stratum] += wi * wi
                odds = 1.0
                p = 0.0
                if has_system[c]:
                    eta = 0.0
                    for a in range(K):
                        eta += x[i, t, a] * beta[c, a]
                    raw_p = _expit(eta)
                    p = min(raw_p, CAP)
                    odds = (p / (1.0 - p)) if p < trim else 0.0
                    if raw_p > CAP:
                        ncap += 1
                    if (not treated) and p >= trim:
                        ntrim += 1
                    v = wi * p * (1.0 - p)
                    for a in range(K):
                        xa = x[i, t, a] * v
                        for b in range(K):
                            a_h[a, b] += xa * x[i, t, b]
                    lo = min(lo, p)
                    hi = max(hi, p)
                m0 = m1 = m2 = m3 = 0.0
                if method != IPW:
                    for a in range(K):
                        xa = x[i, t, a]
                        m0 += xa * models[c, 0, a]
                        m1 += xa * models[c, 1, a]
                        if method == DR:
                            m2 += xa * models[c, 2, a]
                            m3 += xa * models[c, 3, a]
                nt = _row_terms(method, treated, post, odds, y[i, t], m0, m1, m2, m3, Wt, Vt)
                for k in range(nt):
                    wk = Wt[k] * wi
                    wv = wk * Vt[k]
                    a_w[k] += wk
                    a_v[k] += wv
                    for a in range(K):
                        xa = x[i, t, a]
                        a_x[k, a] += wk * xa
                        a_xv[k, a] += wv * xa
                if has_system[c]:
                    # the control terms' weights are the last two ipw/dr terms
                    a_c[0] += (Wt[3] * wi) ** 2
                    a_c[1] += (Wt[2] * wi) ** 2
        weight[c, :] = a_w
        value[c, :] = a_v
        xw[c, :, :] = a_x
        xv[c, :, :] = a_xv
        hess[c, :, :] = a_h
        strata[c, :] = a_n
        sw[c, :] = a_s
        sw2[c, :] = a_s2
        trimmed[c] = ntrim
        capped[c] = ncap
        cws[c, :] = a_c
        pmin[c] = lo
        pmax[c] = hi


@njit(parallel=True, cache=True)
def row_scores(
    g,
    y,
    x,
    w2,
    members,
    cohort,
    dates,
    method,
    trim,
    has_system,
    beta,
    models,
    n_models,
    correction,
    ps_correction,
    mean,
    total,
    signs,
    out,
):
    """The unit scores of each comparison; ``out[c, unit]`` (``RowFunctional.scores``)."""
    n = g.shape[0]
    K = x.shape[2]
    for c in prange(members.shape[0]):
        Wt = np.zeros(8)
        Vt = np.zeros(8)
        for i in range(n):
            if members[c, g[i]] == 0:
                continue
            treated = g[i] == cohort[c]
            acc = 0.0
            for pi in range(2):
                t = dates[c, pi]
                if not np.isfinite(y[i, t]):
                    continue
                post = pi == 1
                wi = _weight(w2, i, t)
                odds = 1.0
                p = 0.0
                if has_system[c]:
                    eta = 0.0
                    for a in range(K):
                        eta += x[i, t, a] * beta[c, a]
                    p = min(_expit(eta), CAP)
                    odds = (p / (1.0 - p)) if p < trim else 0.0
                m0 = m1 = m2 = m3 = 0.0
                if method != IPW:
                    for a in range(K):
                        xa = x[i, t, a]
                        m0 += xa * models[c, 0, a]
                        m1 += xa * models[c, 1, a]
                        if method == DR:
                            m2 += xa * models[c, 2, a]
                            m3 += xa * models[c, 3, a]
                nt = _row_terms(method, treated, post, odds, y[i, t], m0, m1, m2, m3, Wt, Vt)
                phi = 0.0
                for k in range(nt):
                    phi += signs[c, k] * (Wt[k] * wi) * (Vt[k] - mean[c, k]) / total[c, k]
                stratum = (2 if treated else 0) + (1 if post else 0)
                if stratum < n_models[c]:
                    fitted = 0.0
                    adjust = 0.0
                    for a in range(K):
                        fitted += x[i, t, a] * models[c, stratum, a]
                        adjust += x[i, t, a] * correction[c, stratum, a]
                    phi += wi * (y[i, t] - fitted) * adjust
                if has_system[c]:
                    adjust = 0.0
                    for a in range(K):
                        adjust += x[i, t, a] * ps_correction[c, a]
                    phi += wi * ((1.0 if treated else 0.0) - p) * adjust
                acc += phi
            out[c, i] = acc


def fit_linear_rows(batches, keys, size, K):
    """Gram matrices and moments of the strata ``keys`` = [(cohorts, date)]."""
    members = np.zeros((len(keys), size), dtype=np.uint8)
    for j, (cohorts, _) in enumerate(keys):
        members[j, list(cohorts)] = 1
    dates = np.array([t for _, t in keys], dtype=np.int64)
    gram = np.zeros((len(keys), K, K))
    rhs = np.zeros((len(keys), K))
    for raw in batches():
        g, y, x, w2 = RowKernels.arrays(raw)
        q = np.zeros_like(gram)
        r = np.zeros_like(rhs)
        row_linear(g, y, x, w2, members, dates, q, r)
        gram += q
        rhs += r
    return gram, rhs


class RowKernels:
    """Compiled passes for a set of ``cs_unbalanced.RowComparison`` objects."""

    def __init__(self, comparisons, size):
        self.comparisons = comparisons
        self.C = len(comparisons)
        members = np.zeros((self.C, size), dtype=np.uint8)
        for j, c in enumerate(comparisons):
            members[j, [c.cohort, *c.controls]] = 1
        self.members = members
        self.cohort = np.array([c.cohort for c in comparisons], dtype=np.int64)
        self.dates = np.array([[c.base, c.target] for c in comparisons], dtype=np.int64)
        # the interface fit_systems expects of a balanced-panel kernel set
        self.sizes = np.ones(self.C, dtype=np.int64)
        self.Lmax = 1

    @staticmethod
    def arrays(raw):
        g, y, x, w = raw
        g = np.ascontiguousarray(g, dtype=np.int64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        x = np.ascontiguousarray(x, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)
        return g, y, x, np.ascontiguousarray(w if w.ndim == 2 else w[:, None])

    # ---------------------------------------------------------------- logit fits
    def initial(self, raw, K, need_reg, out):
        g, y, x, w2 = self.arrays(raw)
        C = self.C
        counts = np.zeros((C, 2), dtype=np.int64)
        wsums = np.zeros((C, 4))
        scale = np.zeros((C, K))
        row_initial(g, y, x, w2, self.members, self.cohort, self.dates, counts, wsums, scale)
        out["counts"] += counts
        out["wsums"] += wsums
        out["scale"] += scale

    def newton(self, raw, betas, active, K):
        g, y, x, w2 = self.arrays(raw)
        C = self.C
        loss = np.zeros(C)
        grad = np.zeros((C, K))
        hess = np.zeros((C, K, K))
        margin = np.full(C, np.inf)
        row_newton(
            g, y, x, w2, self.members, self.cohort, self.dates, betas, active, loss, grad, hess,
            margin,
        )  # fmt: skip
        return loss, grad, hess, margin

    # ---------------------------------------------------------------- the rest
    def parameters(self, fitted, method, K):
        """Arrays of the fitted comparisons' parameters for the moment and score passes."""
        C = self.C
        has_system = np.array([f.system is not None for f in fitted], dtype=bool)
        beta = np.zeros((C, K))
        models = np.zeros((C, 4, K))
        for j, f in enumerate(fitted):
            if f.system is not None:
                beta[j] = f.system.beta
            for k, (b, _) in enumerate(f.models):
                models[j, k] = b
        return has_system, beta, models

    def moments(self, raw, method, trim, params, acc):
        g, y, x, w2 = self.arrays(raw)
        K = x.shape[2]
        has_system, beta, models = params
        part = self.empty_moments(K)
        row_moments(
            g, y, x, w2, self.members, self.cohort, self.dates, METHOD_CODES[method], trim,
            has_system, beta, models, part["weight"], part["value"], part["xw"], part["xv"],
            part["hess"], part["strata"], part["sw"], part["sw2"], part["trimmed"],
            part["capped"], part["cws"], part["pmin"], part["pmax"],
        )  # fmt: skip
        for key in part:
            if key == "pmin":
                acc[key] = np.minimum(acc[key], part[key])
            elif key == "pmax":
                acc[key] = np.maximum(acc[key], part[key])
            else:
                acc[key] += part[key]

    def empty_moments(self, K):
        C = self.C
        return dict(
            weight=np.zeros((C, 8)),
            value=np.zeros((C, 8)),
            xw=np.zeros((C, 8, K)),
            xv=np.zeros((C, 8, K)),
            hess=np.zeros((C, K, K)),
            strata=np.zeros((C, 4), dtype=np.int64),
            sw=np.zeros((C, 4)),
            sw2=np.zeros((C, 4)),
            trimmed=np.zeros(C, dtype=np.int64),
            capped=np.zeros(C, dtype=np.int64),
            cws=np.zeros((C, 2)),
            pmin=np.full(C, np.inf),
            pmax=np.full(C, -np.inf),
        )

    def scores(self, raw, method, trim, params, fitted):
        """Scores of a batch, units by comparisons."""
        g, y, x, w2 = self.arrays(raw)
        C, K = self.C, x.shape[2]
        has_system, beta, models = params
        n_models = np.array([len(f.models) for f in fitted], dtype=np.int64)
        correction = np.zeros((C, 4, K))
        ps_correction = np.zeros((C, K))
        mean = np.zeros((C, 8))
        total = np.ones((C, 8))
        signs = np.zeros((C, 8))
        for j, f in enumerate(fitted):
            for k, corr in enumerate(f.reg_correction):
                correction[j, k] = corr
            if f.system is not None:
                ps_correction[j] = f.ps_correction
            mean[j, : len(f.signs)] = f.means.mean
            total[j, : len(f.signs)] = f.means.weight
            signs[j, : len(f.signs)] = f.signs
        out = np.zeros((C, len(g)))
        row_scores(
            g, y, x, w2, self.members, self.cohort, self.dates, METHOD_CODES[method], trim,
            has_system, beta, models, n_models, correction, ps_correction, mean, total, signs, out,
        )  # fmt: skip
        return out.T
