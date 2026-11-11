"""A dense, direct transcription of DIDmultiplegtDYN 2.4.0's estimator (no covariates). Never production code.

Missing periods (NaN outcome and treatment) allowed; unit weights, no clustering, no trends.
Y and D are groups x periods. Follows did_multiplegt_main / did_multiplegt_dyn_core /
compute_placebo_effects_polars / compute_E_hat_gt_polars / compute_DOF_gt_polars.
"""

import numpy as np


def prepare(D, *, drop_larger_lower=True):
    """Groups, baselines and windows from the treatment, as did_multiplegt_main.

    Missing periods (NaN) are periods with neither outcome nor treatment.
    """
    G, T = D.shape
    t = np.arange(1, T + 1)
    obs = np.isfinite(D)
    first_obs = obs.argmax(axis=1)
    last_obs = T - 1 - obs[:, ::-1].argmax(axis=1)
    keep = obs.any(axis=1)
    d_sq = D[np.arange(G), first_obs].copy()
    diff = np.where(obs, D - d_sq[:, None], 0.0)
    if drop_larger_lower:
        keep &= ~((diff > 0).any(axis=1) & (diff < 0).any(axis=1))
    changed = obs & (np.abs(diff) > 0)
    ever = np.maximum.accumulate(changed, axis=1)
    F = np.where(ever.any(axis=1), ever.argmax(axis=1) + 1, 0)  # periods 1..T, 0 never
    # design restriction: within a baseline, F must vary (never-switchers count as F=0)
    for d in np.unique(d_sq[keep]):
        mine = keep & (d_sq == d)
        if np.unique(F[mine]).size < 2:
            keep[mine] = False
    # (t, d) rows where no observed group of baseline d is unswitched are dropped
    present = obs.copy()
    for d in np.unique(d_sq[keep]):
        mine = keep & (d_sq == d)
        open_ = (obs[mine] & ~ever[mine]).any(axis=0)
        present[np.ix_(mine, ~open_)] = False
    G_R = int(keep.sum())  # G as DIDmultiplegtDYN counts it
    T_max = T
    # the last observed period before the switch; a gap just before the switch makes
    # its date uncertain: the group is a control until the gap, and nothing after
    before = obs & (t[None, :] < np.where(F > 0, F, T + 1)[:, None])
    last_before = np.where(before.any(axis=1), T - before[:, ::-1].argmax(axis=1), 0)
    gap = (F > 0) & (last_before < F - 1)
    trunc = np.where(gap, last_before + 1, np.where(F == 0, last_obs + 2, 0))
    present &= ~(gap[:, None] & (t[None, :] > last_before[:, None]))
    F = np.where((F == 0) | gap, T_max + 1, F)
    F_trunc = np.where(trunc > 0, np.minimum(F, trunc), F)
    T_g = np.zeros(G, dtype=int)
    for d in np.unique(d_sq[keep]):
        mine = keep & (d_sq == d)
        T_g[mine] = F_trunc[mine].max() - 1
    window = (t[None, :] >= F[:, None]) & (t[None, :] <= T_g[:, None]) & present
    with np.errstate(invalid="ignore", divide="ignore"):
        avg = np.where(window, D, 0).sum(axis=1) / window.sum(axis=1)
    back = (avg == d_sq) & np.isfinite(avg) & (F != T_g + 1)
    keep &= ~back
    S = np.where(F != T_max + 1, (avg > d_sq).astype(float), np.nan)
    d_fg = np.array([D[g, F[g] - 1] if F[g] <= T else d_sq[g] for g in range(G)])
    L = T_g - F + 1
    return dict(
        keep=keep,
        d_sq=d_sq,
        F=F,
        T_g=T_g,
        S=S,
        d_fg=d_fg,
        L=L,
        present=present,
        T_max=T_max,
        G_R=G_R,
    )


def covariate_stats(Y, X, info):
    """Per baseline: the coefficients of the outcome's one-period changes on the
    covariates' among not-yet-switched rows, with period effects (did_multiplegt_main),
    and each group's term of the correction for estimating them."""
    keep, d_sq, F, present = (info[k] for k in ("keep", "d_sq", "F", "present"))
    G, T, K = X.shape
    t = np.arange(1, T + 1)[None, :].repeat(G, axis=0)
    dY = np.full((G, T), np.nan)
    dY[:, 1:] = Y[:, 1:] - Y[:, :-1]
    dX = np.full((G, T, K), np.nan)
    dX[:, 1:] = X[:, 1:] - X[:, :-1]
    rows_all = keep[:, None] & present & (t >= 2) & (t < F[:, None])
    rows_all &= np.isfinite(dY) & np.isfinite(dX).all(axis=2)
    theta, inv_denom = {}, {}
    in_sum = np.zeros((G, K))
    for d in np.unique(d_sq[keep]):
        rows = rows_all & (d_sq == d)[:, None]
        n = rows.sum(axis=0)
        avg = np.zeros((T, K))
        for tt in range(T):
            if n[tt]:
                avg[tt] = dX[rows[:, tt], tt].mean(axis=0)
        resid = np.where(rows[:, :, None], dX - avg[None], 0.0)
        M = np.einsum("gtk,gtj->kj", resid, resid)
        v = np.einsum("gtk,gt->k", resid, np.where(rows, dY, 0.0))
        invM = np.linalg.inv(M)
        th = invM @ v
        fitted = np.einsum("gtk,k->gt", np.nan_to_num(dX), th)
        fe = np.zeros(T)
        for tt in range(T):
            if n[tt]:
                fe[tt] = (dY[rows[:, tt], tt] - fitted[rows[:, tt], tt]).mean()
        E = np.where(n[None, :] >= 2, fitted + fe[None, :], 0.0)
        N_c = rows.sum()
        factor = np.where(n >= 2, np.sqrt(n / np.maximum(n - 1, 1)), 1.0)[None, :]
        term = np.where(rows, factor * (np.nan_to_num(dY) - E) / N_c, 0.0)
        in_sum += np.einsum("gtk,gt->gk", resid, term)
        theta[d] = th
        inv_denom[d] = invM * N_c * info["G_R"]
    return dict(theta=theta, inv_denom=inv_denom, in_sum=in_sum, X=X)


def _cell_mean(values, weights, mask, keys):
    """Sum of weights and of weights*values by key over masked rows; per-row mean and row count."""
    out_mean = np.full(values.shape, np.nan)
    out_count = np.full(values.shape, np.nan)
    flat = [tuple(k) for k in keys.reshape(-1, keys.shape[-1])]
    m, v, w = mask.ravel(), values.ravel(), weights.ravel()
    sums, wsum, n = {}, {}, {}
    for i, k in enumerate(flat):
        if m[i]:
            sums[k] = sums.get(k, 0.0) + w[i] * v[i]
            wsum[k] = wsum.get(k, 0.0) + w[i]
            n[k] = n.get(k, 0) + 1
    om, oc = out_mean.ravel(), out_count.ravel()
    for i, k in enumerate(flat):
        if k in wsum:
            om[i] = sums[k] / wsum[k]
            oc[i] = n[k] if m[i] else np.nan
    return out_mean, out_count


def _one(Y, D, info, ell, direction, kind, only_never, cov=None):
    """U and U_var by group for effect (kind='effect') or placebo ell, one direction."""
    keep, d_sq, F, T_g, S, d_fg, L, present = (
        info[k] for k in ("keep", "d_sq", "F", "T_g", "S", "d_fg", "L", "present")
    )
    G, T = Y.shape
    T_max = info["T_max"]
    t = np.arange(1, T + 1)[None, :].repeat(G, axis=0)
    Yp = np.where(present & keep[:, None], Y, np.nan)
    N_gt = np.where(np.isfinite(Yp), 1.0, 0.0)

    def lag(a, k):
        out = np.full(a.shape, np.nan)
        if k < T:
            out[:, k:] = a[:, :-k] if k else a
        return out

    diff = Yp - lag(Yp, ell)
    never = np.where(np.isfinite(diff), (F[:, None] > t).astype(float), np.nan)
    if only_never:
        never = np.where(
            np.isfinite(diff) & (F[:, None] > t) & (F[:, None] < T_max + 1), 0.0, never
        )
    base = np.repeat(d_sq[:, None], T, axis=1)
    keys_dt = np.stack([base, t.astype(float)], axis=-1)

    def by_dt(values):
        out = np.zeros((G, T))
        for d in np.unique(d_sq[keep]):
            mine = keep & (d_sq == d)
            out[mine] = np.nansum(values[mine], axis=0)
        return out

    N_ctrl = by_dt(np.where(keep[:, None], never * N_gt, np.nan))
    sw = (
        (t == F[:, None] - 1 + ell) & (ell <= L[:, None]) & (S[:, None] == direction) & (N_ctrl > 0)
    )
    dist = np.where(np.isfinite(diff), sw.astype(float), np.nan)
    dist = np.where(keep[:, None], dist, np.nan)
    if kind == "placebo":
        dpl = lag(Yp, 2 * ell) - lag(Yp, ell)
        never = never * np.isfinite(dpl)
        never = np.where(np.isfinite(dpl), never, np.where(np.isfinite(never), 0.0, np.nan))
        N_ctrl = by_dt(np.where(keep[:, None], never * N_gt, np.nan))
        dist = np.where(np.isfinite(dist), dist * np.isfinite(dpl) * (N_ctrl > 0), np.nan)
        diff = dpl
    if cov is not None:
        # residualize the changes with the baseline's coefficients
        Xp = np.where((present & keep[:, None])[:, :, None], cov["X"], np.nan)
        lagX = np.full(Xp.shape, np.nan)
        lagX[:, ell:] = Xp[:, :-ell]
        if kind == "effect":
            dXl = Xp - lagX
        else:
            lag2 = np.full(Xp.shape, np.nan)
            if 2 * ell < T:
                lag2[:, 2 * ell :] = Xp[:, : -2 * ell]
            dXl = lag2 - lagX
        th = np.array([cov["theta"].get(d, np.zeros(Xp.shape[2])) for d in d_sq])
        diff = diff - np.einsum("gtk,gk->gt", dXl, th)
    N = np.nansum(dist * N_gt)
    if N == 0:
        return 0.0, None, None
    N_sw = by_dt(np.where(keep[:, None], dist * N_gt, np.nan))
    N_t = np.nansum(np.where(keep[:, None], dist * N_gt, np.nan), axis=0)[None, :].repeat(G, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(N_ctrl == 0, 0.0, N_sw / N_ctrl)
    rows = (t >= ell + 1) & (t <= T_g[:, None])
    dummy = (ell <= T_g - 1).astype(float)[:, None]
    weight = dummy * rows * N_gt * (dist - ratio * never)
    U = np.nansum(np.where(keep[:, None], weight * diff, np.nan), axis=1) / N  # U/G in R's scale
    # variance
    dof_ns = (N_gt != 0) & np.isfinite(diff) & (never == 1) & (N_t > 0)
    dof_s = (N_gt != 0) & (dist == 1)
    keys_s = np.stack(
        [base, np.repeat(F[:, None].astype(float), T, 1), np.repeat(d_fg[:, None], T, 1)], axis=-1
    )
    m_ns, c_ns = _cell_mean(np.nan_to_num(diff), N_gt, dof_ns & keep[:, None], keys_dt)
    m_s, c_s = _cell_mean(np.nan_to_num(diff), N_gt, dof_s & keep[:, None], keys_s)
    union = (dof_s | dof_ns) & keep[:, None]
    m_u, c_u = _cell_mean(np.nan_to_num(diff), N_gt, union, keys_dt)
    pre = t < F[:, None]
    at = t == F[:, None] - 1 + ell
    E = np.where(pre | at, 0.0, np.nan)
    E = np.where(pre & (c_ns >= 2), m_ns, E)
    E = np.where(at & (c_s >= 2), m_s, E)
    E = np.where((c_u >= 2) & ((at & (c_s == 1)) | (pre & (c_ns == 1))), m_u, E)
    DOF = np.where(pre | at, 1.0, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        DOF = np.where(at & (c_s > 1), np.sqrt(c_s / (c_s - 1)), DOF)
        DOF = np.where(pre & (c_ns > 1), np.sqrt(c_ns / (c_ns - 1)), DOF)
        DOF = np.where(
            (c_u >= 2) & ((at & (c_s == 1)) | (pre & (c_ns == 1))), np.sqrt(c_u / (c_u - 1)), DOF
        )
    Uvar = np.nansum(np.where(keep[:, None], weight * DOF * (diff - E), np.nan), axis=1) / N
    if cov is not None:
        # the correction for the estimated coefficients (part2 in did_multiplegt_dyn_core)
        strict = (ell <= T_g - 2).astype(float)[:, None]
        part2 = np.zeros(G)
        for d, th in cov["theta"].items():
            mine = (d_sq == d) & keep
            Mj = (
                np.nansum(
                    np.where(mine[:, None, None], (strict * weight)[:, :, None] * dXl, np.nan),
                    axis=(0, 1),
                )
                / N
            )
            own = (mine & (F >= 3))[:, None] * (cov["in_sum"] @ cov["inv_denom"][d].T)
            part2 += (own - th[None, :]) @ Mj
        Uvar = np.where(keep, Uvar - part2 / info["G_R"], Uvar)
    # normalization: average cumulative treatment change of the switchers
    dev = np.where(
        (t >= F[:, None]) & (t <= F[:, None] - 1 + ell), np.nan_to_num(D - d_sq[:, None]), 0
    ).sum(axis=1)
    sign = np.where(S == 1, dev, -dev)
    delta = np.nansum(np.where(dist == 1, N_gt * sign[:, None], 0)) / N
    # average total effect: treatment change at this horizon only
    at_dev = np.where(
        dist == 1, N_gt * np.where(S[:, None] == 1, D - d_sq[:, None], d_sq[:, None] - D), 0
    )
    delta_now = np.nansum(at_dev) / N
    return N, (U, Uvar), (delta, delta_now)


def did_multiplegt_dyn(
    Y,
    D,
    effects,
    placebo=0,
    *,
    normalized=False,
    only_never_switchers=False,
    switchers="both",
    X=None,
):
    directions = {"both": (1, 0), "in": (1,), "out": (0,)}[switchers]
    info = prepare(D)
    cov = covariate_stats(Y, X, info) if X is not None else None
    G = Y.shape[0]
    out = {}
    L_in = np.nanmax(np.where(info["keep"] & (info["S"] == 1), info["L"], -np.inf))
    L_out = np.nanmax(np.where(info["keep"] & (info["S"] == 0), info["L"], -np.inf))
    n_eff = int(min(max(L for L, s in ((L_in, 1), (L_out, 0)) if s in directions), effects))
    F, L, S, keep = info["F"], info["L"], info["S"], info["keep"]
    span = np.where(F >= 3, np.minimum(L, F - 2), 0)
    P_max = max((span[keep & (S == s)].max(initial=0) for s in directions), default=0)
    for kind, count in (("effect", n_eff), ("placebo", int(min(placebo, P_max, n_eff)))):
        for ell in range(1, count + 1):
            parts = {
                s: _one(Y, D, info, ell, s, kind, only_never_switchers, cov)
                if s in directions
                else (0.0, None, None)
                for s in (1, 0)
            }
            N1, N0 = parts[1][0], parts[0][0]
            total = N1 + N0
            U = np.zeros(G)
            Uvar = np.zeros(G)
            delta = 0.0
            for s, sign in ((1, 1.0), (0, -1.0)):
                Ns, scores, deltas = parts[s]
                if Ns:
                    U += Ns / total * sign * scores[0]
                    Uvar += Ns / total * sign * scores[1]
                    delta += Ns / total * deltas[0]
            est, se = U.sum(), np.sqrt((Uvar**2).sum())
            if normalized:
                est, se, Uvar = est / delta, se / delta, Uvar / delta
            out[(kind, ell)] = dict(estimate=est, se=se, n=total, Uvar=Uvar, parts=parts)
    # average total effect
    num, den, var = {}, {}, {}
    for s, sign in ((1, 1.0), (0, -1.0)):
        Ns = [out[("effect", ell)]["parts"][s] for ell in range(1, n_eff + 1)]
        total = sum(p[0] for p in Ns)
        if not total:
            continue
        w = [p[0] / total for p in Ns]
        num[s] = sum(wi * sign * p[1][0] for wi, p in zip(w, Ns, strict=True) if p[0])
        var[s] = sum(wi * sign * p[1][1] for wi, p in zip(w, Ns, strict=True) if p[0])
        den[s] = sum(wi * p[2][1] for wi, p in zip(w, Ns, strict=True) if p[0])
        den[s] = (den[s], total)
    if num:
        dplus = den.get(1, (0, 0))
        dminus = den.get(0, (0, 0))
        denom = dplus[0] * dplus[1] + dminus[0] * dminus[1]
        wp = dplus[0] * dplus[1] / denom if denom > 0 else 0.5
        U = (wp * num[1] / dplus[0] if 1 in num else 0) + (
            (1 - wp) * num[0] / dminus[0] if 0 in num else 0
        )
        V = (wp * var[1] / dplus[0] if 1 in var else 0) + (
            (1 - wp) * var[0] / dminus[0] if 0 in var else 0
        )
        out["average"] = dict(estimate=U.sum(), se=np.sqrt((V**2).sum()), Uvar=V)
    return out
