"""Compiled passes of the Callaway-Sant'Anna covariate fit on balanced panels.

A fit has many comparisons (a cohort against its controls, with the covariates
of one period), and every pass over the staged units visits each unit once per
comparison it belongs to. In NumPy that is a few hundred small array operations
per batch, which the interpreter's lock serializes. The kernels here do the
same sums in one compiled call per batch, in parallel over comparisons.

Each comparison is handled by one thread and adds its rows in the order they
come, so the sums, and with them the estimates, do not depend on the number of
threads. The NumPy code in ``nuisance.py``, ``contrasts.py`` and ``cs.py``
remains the reference (and the path for unbalanced panels); the tests compare
the two.
"""

import numpy as np
from numba import njit, prange

ENABLED = True  # tests switch it off to compare with the array path

REG, IPW, DR = 0, 1, 2
METHOD_CODES = {"reg": REG, "ipw": IPW, "dr": DR}


@njit(cache=True, inline="always")
def _expit(z):
    if z >= 0.0:
        return 1.0 / (1.0 + np.exp(-z))
    e = np.exp(z)
    return e / (1.0 + e)


@njit(parallel=True, cache=True)
def initial_sums(
    g,
    y,
    x,
    w2,
    members,
    cohort,
    xper,
    wper,
    cell_ptr,
    tgt,
    base,
    need_reg,
    counts,
    wsums,
    scale,
    gram,
    reg,
):
    """Counts, weight sums, design scales and the control regression's moments.

    counts[c] = (treated, controls); wsums[c] = (treated weight, control weight,
    treated weight squares, control weight squares).
    """
    n = g.shape[0]
    K = x.shape[2]
    for c in prange(members.shape[0]):
        first = cell_ptr[c]
        L = cell_ptr[c + 1] - first
        xp = xper[c]
        wp = wper[c]
        # local accumulators: neighbouring comparisons' outputs share cache lines
        cnt = np.zeros(2, dtype=np.int64)
        ws = np.zeros(4)
        sc = np.zeros(K)
        gr = np.zeros((K, K))
        rg = np.zeros((K, L))
        for i in range(n):
            if members[c, g[i]] == 0:
                continue
            treated = g[i] == cohort[c]
            wi = w2[i, wp]
            if treated:
                cnt[0] += 1
                ws[0] += wi
                ws[2] += wi * wi
            else:
                cnt[1] += 1
                ws[1] += wi
                ws[3] += wi * wi
            for a in range(K):
                xa = x[i, xp, a]
                sc[a] += wi * xa * xa
            if need_reg and not treated:
                for a in range(K):
                    xa = x[i, xp, a] * wi
                    for b in range(K):
                        gr[a, b] += xa * x[i, xp, b]
                    for k in range(L):
                        rg[a, k] += xa * (y[i, tgt[first + k]] - y[i, base[first + k]])
        counts[c, :] = cnt
        wsums[c, :] = ws
        scale[c, :] = sc
        if need_reg:
            gram[c, :, :] = gr
            reg[c, :, :L] = rg


@njit(parallel=True, cache=True)
def newton_stats(g, x, w2, members, cohort, xper, wper, beta, active, loss, grad, hess, margin):
    """Logit loss, gradient, information and smallest margin of the active comparisons."""
    n = g.shape[0]
    K = x.shape[2]
    for c in prange(members.shape[0]):
        if not active[c]:
            continue
        xp = xper[c]
        wp = wper[c]
        total = 0.0
        low = np.inf
        gd = np.zeros(K)
        hs = np.zeros((K, K))
        for i in range(n):
            if members[c, g[i]] == 0:
                continue
            d = 1.0 if g[i] == cohort[c] else 0.0
            wi = w2[i, wp]
            eta = 0.0
            for a in range(K):
                eta += x[i, xp, a] * beta[c, a]
            e = np.exp(-abs(eta))
            p = 1.0 / (1.0 + e) if eta >= 0.0 else e / (1.0 + e)
            total += wi * (max(eta, 0.0) + np.log1p(e) - d * eta)
            r = wi * (d - p)
            v = wi * p * (1.0 - p)
            for a in range(K):
                xa = x[i, xp, a]
                gd[a] += xa * r
                xv = xa * v
                for b in range(K):
                    hs[a, b] += xv * x[i, xp, b]
            m = (2.0 * d - 1.0) * eta
            if m < low:
                low = m
        loss[c] = total
        grad[c, :] = gd
        hess[c, :, :] = hs
        margin[c] = low


@njit(parallel=True, cache=True)
def contrast_moments(
    g,
    y,
    x,
    w2,
    members,
    cohort,
    xper,
    wper,
    cell_ptr,
    tgt,
    base,
    method,
    beta,
    reg,
    control_max,
    s1,
    xt,
    w0,
    w02,
    s0,
    xc,
    xcy,
    hess,
    pmin,
    pmax,
    trimmed,
    capped,
):
    """The moments of one pass after the nuisance fits (``fit_contrasts``)."""
    n = g.shape[0]
    K = x.shape[2]
    cap = 1.0 - 1e-6
    for c in prange(members.shape[0]):
        first = cell_ptr[c]
        L = cell_ptr[c + 1] - first
        xp = xper[c]
        wp = wper[c]
        res = np.empty(L)
        a1 = np.zeros(L)
        axt = np.zeros(K)
        a0 = np.zeros(L)
        axc = np.zeros(K)
        axcy = np.zeros((K, L))
        ah = np.zeros((K, K))
        sw0 = 0.0
        sw02 = 0.0
        lo = 1.0
        hi = 0.0
        nt = 0
        nc = 0
        for i in range(n):
            if members[c, g[i]] == 0:
                continue
            treated = g[i] == cohort[c]
            wi = w2[i, wp]
            for k in range(L):
                r = y[i, tgt[first + k]] - y[i, base[first + k]]
                if method != IPW:
                    for a in range(K):
                        r -= x[i, xp, a] * reg[c, a, k]
                res[k] = r
            if treated:
                for k in range(L):
                    a1[k] += wi * res[k]
                for a in range(K):
                    axt[a] += wi * x[i, xp, a]
            if method == REG:
                continue
            eta = 0.0
            for a in range(K):
                eta += x[i, xp, a] * beta[c, a]
            p = min(_expit(eta), cap)
            if p < lo:
                lo = p
            if p > hi:
                hi = p
            if (not treated) and p >= control_max:
                nt += 1
            if p >= cap:
                nc += 1
            wc = 0.0
            if (not treated) and p < control_max:
                wc = p / (1.0 - p) * wi
            sw0 += wc
            sw02 += wc * wc
            for k in range(L):
                a0[k] += wc * res[k]
            v = wi * p * (1.0 - p)
            for a in range(K):
                xa = x[i, xp, a]
                axc[a] += wc * xa
                for k in range(L):
                    axcy[a, k] += xa * wc * res[k]
                xv = xa * v
                for b in range(K):
                    ah[a, b] += xv * x[i, xp, b]
        s1[c, :L] = a1
        xt[c, :] = axt
        s0[c, :L] = a0
        xc[c, :] = axc
        xcy[c, :, :L] = axcy
        hess[c, :, :] = ah
        w0[c] = sw0
        w02[c] = sw02
        pmin[c] = lo
        pmax[c] = hi
        trimmed[c] = nt
        capped[c] = nc


@njit(parallel=True, cache=True)
def unit_scores(
    g,
    y,
    x,
    w2,
    members,
    cohort,
    xper,
    wper,
    cell_ptr,
    tgt,
    base,
    col,
    method,
    beta,
    reg,
    control_max,
    treated_mean,
    control_mean,
    treated_total,
    control_total,
    outcome_adj,
    prop_adj,
    scores,
):
    """Each unit's influence-function scores, written into its comparison's columns."""
    n = g.shape[0]
    K = x.shape[2]
    cap = 1.0 - 1e-6
    for c in prange(members.shape[0]):
        first = cell_ptr[c]
        L = cell_ptr[c + 1] - first
        xp = xper[c]
        wp = wper[c]
        for i in range(n):
            if members[c, g[i]] == 0:
                continue
            treated = g[i] == cohort[c]
            wi = w2[i, wp]
            p = 0.0
            wc = 0.0
            if method != REG:
                eta = 0.0
                for a in range(K):
                    eta += x[i, xp, a] * beta[c, a]
                p = min(_expit(eta), cap)
                if (not treated) and p < control_max:
                    wc = p / (1.0 - p)
            oadj = 0.0
            if method != IPW and not treated:
                for a in range(K):
                    oadj += x[i, xp, a] * outcome_adj[c, a]
            dpx = (1.0 if treated else 0.0) - p
            for k in range(L):
                r = y[i, tgt[first + k]] - y[i, base[first + k]]
                if method != IPW:
                    for a in range(K):
                        r -= x[i, xp, a] * reg[c, a, k]
                phi = 0.0
                if treated:
                    phi += (r - treated_mean[c, k]) / treated_total[c]
                if method != REG:
                    phi -= wc * (r - control_mean[c, k]) / control_total[c]
                    adj = 0.0
                    for a in range(K):
                        adj += x[i, xp, a] * prop_adj[c, a, k]
                    phi += dpx * adj
                if method != IPW and not treated:
                    phi += r * oadj
                scores[i, col[first + k]] = phi * wi


class BalancedCS:
    """The compiled passes for a set of comparisons on balanced batches.

    ``comparisons`` are ``cs.Comparison`` objects. ``size`` is the number of
    cohort labels (members tables have that length).
    """

    def __init__(self, comparisons, n_cells):
        self.comparisons = comparisons
        C = len(comparisons)
        self.C = C
        self.members = np.ascontiguousarray(
            np.vstack([c.members for c in comparisons]), dtype=np.uint8
        )
        self.cohort = np.array([c.cohort for c in comparisons], dtype=np.int64)
        self.xper = np.array([c.xperiod for c in comparisons], dtype=np.int64)
        self.wper = np.array([c.weightperiod for c in comparisons], dtype=np.int64)
        sizes = np.array([len(c.cells) for c in comparisons], dtype=np.int64)
        self.cell_ptr = np.r_[0, np.cumsum(sizes)].astype(np.int64)
        self.tgt = np.concatenate([c.targets for c in comparisons]).astype(np.int64)
        self.base = np.concatenate([c.bases for c in comparisons]).astype(np.int64)
        self.col = np.concatenate([np.asarray(c.cells) for c in comparisons]).astype(np.int64)
        self.sizes = sizes
        self.Lmax = int(sizes.max())
        self.n_cells = n_cells

    def arrays(self, raw):
        """(g, y, x, w2): the batch as contiguous arrays, weights as units by periods."""
        g, y, x, w = raw
        g = np.ascontiguousarray(g, dtype=np.int64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        x = np.ascontiguousarray(x, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)
        w2 = np.ascontiguousarray(w if w.ndim == 2 else w[:, None])
        return g, y, x, w2

    def _wper(self, w2):
        return self.wper if w2.shape[1] > 1 else np.zeros(self.C, dtype=np.int64)

    # ---------------------------------------------------------------- nuisance
    def initial(self, raw, K, need_reg, out):
        """Add a batch's counts, weights, scales and control moments to ``out``."""
        g, y, x, w2 = self.arrays(raw)
        C = self.C
        counts = np.zeros((C, 2), dtype=np.int64)
        wsums = np.zeros((C, 4))
        scale = np.zeros((C, K))
        gram = np.zeros((C, K, K)) if need_reg else np.zeros((1, 1, 1))
        reg = np.zeros((C, K, self.Lmax)) if need_reg else np.zeros((1, 1, 1))
        initial_sums(
            g,
            y,
            x,
            w2,
            self.members,
            self.cohort,
            self.xper,
            self._wper(w2),
            self.cell_ptr,
            self.tgt,
            self.base,
            need_reg,
            counts,
            wsums,
            scale,
            gram,
            reg,
        )
        out["counts"] += counts
        out["wsums"] += wsums
        out["scale"] += scale
        if need_reg:
            out["gram"] += gram
            out["reg"] += reg

    def newton(self, raw, betas, active, K):
        """(loss, gradient, information, margin) of a batch for the active comparisons."""
        g, _, x, w2 = self.arrays(raw)
        C = self.C
        loss = np.zeros(C)
        grad = np.zeros((C, K))
        hess = np.zeros((C, K, K))
        margin = np.full(C, np.inf)
        newton_stats(
            g,
            x,
            w2,
            self.members,
            self.cohort,
            self.xper,
            self._wper(w2),
            betas,
            active,
            loss,
            grad,
            hess,
            margin,
        )
        return loss, grad, hess, margin

    # ---------------------------------------------------------------- contrasts
    def moments(self, raw, method, beta, reg, control_max, K, acc):
        g, y, x, w2 = self.arrays(raw)
        C, L = self.C, self.Lmax
        part = dict(
            s1=np.zeros((C, L)),
            xt=np.zeros((C, K)),
            w0=np.zeros(C),
            w02=np.zeros(C),
            s0=np.zeros((C, L)),
            xc=np.zeros((C, K)),
            xcy=np.zeros((C, K, L)),
            hess=np.zeros((C, K, K)),
            pmin=np.ones(C),
            pmax=np.zeros(C),
            trimmed=np.zeros(C, dtype=np.int64),
            capped=np.zeros(C, dtype=np.int64),
        )
        contrast_moments(
            g,
            y,
            x,
            w2,
            self.members,
            self.cohort,
            self.xper,
            self._wper(w2),
            self.cell_ptr,
            self.tgt,
            self.base,
            METHOD_CODES[method],
            beta,
            reg,
            control_max,
            part["s1"],
            part["xt"],
            part["w0"],
            part["w02"],
            part["s0"],
            part["xc"],
            part["xcy"],
            part["hess"],
            part["pmin"],
            part["pmax"],
            part["trimmed"],
            part["capped"],
        )
        for key in ("s1", "xt", "w0", "w02", "s0", "xc", "xcy", "hess", "trimmed", "capped"):
            acc[key] += part[key]
        acc["pmin"] = np.minimum(acc["pmin"], part["pmin"])
        acc["pmax"] = np.maximum(acc["pmax"], part["pmax"])

    # ---------------------------------------------------------------- scores
    def scores(self, raw, method, packed, control_max):
        """The batch's scores, units by cells; ``packed`` as ``pack_fitted`` makes it."""
        g, y, x, w2 = self.arrays(raw)
        out = np.zeros((len(g), self.n_cells))
        unit_scores(
            g,
            y,
            x,
            w2,
            self.members,
            self.cohort,
            self.xper,
            self._wper(w2),
            self.cell_ptr,
            self.tgt,
            self.base,
            self.col,
            METHOD_CODES[method],
            packed["beta"],
            packed["reg"],
            control_max,
            packed["treated_mean"],
            packed["control_mean"],
            packed["treated_total"],
            packed["control_total"],
            packed["outcome_adj"],
            packed["prop_adj"],
            out,
        )
        return out

    def pack_fitted(self, fitted, K):
        """Arrays of the fitted contrasts' parameters, padded to the widest comparison."""
        C, L = self.C, self.Lmax
        packed = dict(
            beta=np.zeros((C, K)),
            reg=np.zeros((C, K, L)),
            treated_mean=np.zeros((C, L)),
            control_mean=np.zeros((C, L)),
            treated_total=np.ones(C),
            control_total=np.ones(C),
            outcome_adj=np.zeros((C, K)),
            prop_adj=np.zeros((C, K, L)),
        )
        for j, f in enumerate(fitted):
            n = self.sizes[j]
            packed["beta"][j] = f.nuisance.beta
            packed["reg"][j, :, :n] = f.nuisance.regression
            packed["treated_mean"][j, :n] = f.treated_mean
            packed["control_mean"][j, :n] = f.control_mean
            packed["treated_total"][j] = f.treated_total
            packed["control_total"][j] = f.control_total
            packed["outcome_adj"][j] = f.outcome_adjustment
            packed["prop_adj"][j, :, :n] = f.propensity_adjustment
        return packed
