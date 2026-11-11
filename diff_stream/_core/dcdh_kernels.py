"""Group scores of de Chaisemartin and D'Haultfoeuille's DID_l, one batch of groups at a time.

Indices: slot j (an estimate's horizon: kind 0 effect or 1 placebo, horizon l, and
which switchers it uses), direction s (1 for switchers in, 0 out), cell b (a
first-period treatment, within a trend group), period t (1..T). For each group the
kernel returns its sums, before division by the number of switchers, of

- U: the terms whose sum over groups is the estimate, and
- V: the same with each change demeaned and rescaled as in the authors' variance
  (``compute_E_hat_gt`` and ``compute_DOF_gt`` in DIDmultiplegtDYN),

for every (j, s). Each term carries the group's weight in its own period.
"""

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _demeaned(change, mean, count, union_mean, union_count):
    """(change - E) * DOF for a row in a cohort of ``count`` rows, as DIDmultiplegtDYN."""
    if count >= 2:
        return (change - mean) * np.sqrt(count / (count - 1.0))
    if count == 1 and union_count >= 2:
        return (change - union_mean) * np.sqrt(union_count / (union_count - 1.0))
    return change


@njit(cache=True, parallel=True)
def group_terms(
    base,
    first,
    up,
    cohort,
    w,
    y,
    T_d,
    only_never,
    n_periods,
    slot_kind,
    slot_horizon,
    eligible,
    ctrl_w,
    ctrl_mean,
    ctrl_n,
    sw_w,
    sw_n,
    t_any,
    union_mean,
    union_n,
    cohort_mean,
    cohort_n,
):
    B = y.shape[0]
    S = slot_kind.shape[0]
    U = np.zeros((B, S, 2))
    V = np.zeros((B, S, 2))
    for i in prange(B):
        b = base[i]
        if b < 0:
            continue
        F, Td = first[i], T_d[b]
        for j in range(S):
            k, li = slot_kind[j], slot_horizon[j]
            ell = li + 1
            # as a switcher: the change from F - 1 to F - 1 + l (placebo: to F - 1 - l)
            if up[i] >= 0 and cohort[i] >= 0 and eligible[i, j]:
                t = F - 1 + ell
                s = up[i]
                if t <= Td and sw_w[j, s, b, t] > 0 and (k == 0 or F - 1 - ell >= 1):
                    if k == 0:
                        change = y[i, t - 1] - y[i, F - 2]
                    else:
                        change = y[i, F - 2 - ell] - y[i, F - 2]
                    # a change needs both of its periods observed; a placebo's
                    # switchers also need the effect's change (y at F - 1 + l)
                    if np.isfinite(change) and np.isfinite(y[i, t - 1]):
                        c = cohort[i]
                        wi = w[i, t - 1]
                        U[i, j, s] += wi * change
                        V[i, j, s] += wi * _demeaned(
                            change,
                            cohort_mean[j, c],
                            cohort_n[j, c],
                            union_mean[j, s, b, t],
                            union_n[j, s, b, t],
                        )
            # as a control: every period it has not switched by, for both directions
            if only_never and F <= n_periods:
                continue
            low = ell + 1 if k == 0 else 2 * ell + 1
            high = min(F - 1, Td)
            for t in range(low, high + 1):
                if ctrl_w[k, li, b, t] <= 0:
                    continue
                if k == 0:
                    change = y[i, t - 1] - y[i, t - 1 - ell]
                else:
                    change = y[i, t - 1 - 2 * ell] - y[i, t - 1 - ell]
                if not np.isfinite(change):
                    continue
                if k == 1 and not np.isfinite(y[i, t - 1]):
                    continue  # a placebo's controls also need the effect's change
                wi = w[i, t - 1]
                for s in range(2):
                    n_sw = sw_w[j, s, b, t]
                    if n_sw <= 0 or not t_any[j, s, t]:
                        continue
                    r = wi * n_sw / ctrl_w[k, li, b, t]
                    U[i, j, s] -= r * change
                    V[i, j, s] -= r * _demeaned(
                        change,
                        ctrl_mean[k, li, b, t],
                        ctrl_n[k, li, b, t],
                        union_mean[j, s, b, t],
                        union_n[j, s, b, t],
                    )
    return U, V
