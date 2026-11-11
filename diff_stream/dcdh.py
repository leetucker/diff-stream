"""de Chaisemartin and D'Haultfoeuille's event-study estimator, DID_l (did_multiplegt_dyn).

For a treatment that may take several values and switch on and off, each group's
baseline is its first-period treatment and its first switch the first period its
treatment differs from that. Effect l compares the outcome change of the groups that
switched l periods ago, since the period before their switch, with that of the groups
with the same baseline that have not switched yet; placebo l compares the change from
the period before the switch back l periods. Switchers in (whose treatment rose) and
switchers out (fell) are estimated separately and combined, the latter with the sign
reversed. The conventions follow the authors' R package DIDmultiplegtDYN 2.4.0.

The panel is staged once (each group labeled by its first switch, its treatment path
kept as a column), then read three times: the baselines and their switch periods; the
sums by baseline and period and by switching cohort; and each group's scores.
"""

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
import pyarrow as pa
from hdfe_stream.core.profile import Component, Profile
from hdfe_stream.core.results_base import RegressionResult
from stream_limits import MemoryBudget, current_rss_gb, detect_cpus, thread_scope

from . import _docs, _options
from ._core.dcdh_kernels import group_terms
from ._core.partition import preprocess_budget_mb, staging_bytes
from ._present import public_profile
from .store import PanelStore, inspect

PUBLIC = "did_dcdh"
# Distinct first-period treatment values: each needs a row of every cell table.
MAX_BASELINES = 10_000


@dataclass(kw_only=True)
class DCDHResult(RegressionResult):
    """Dynamic effects of a treatment that may switch on and off and take several values.

    The coefficients are ``effect::l`` (the effect l periods after the first switch),
    ``placebo::l`` and ``average_total_effect``, with their joint covariance.
    ``pretrend_test()`` tests that the placebos are jointly zero,
    ``effects_equal_test()`` that the effects are equal, and ``switchers()`` lists the
    groups behind each estimate.
    """

    n_units: int
    _switchers: pl.DataFrame = field(repr=False)
    _internal: dict | None = field(default=None, repr=False)
    estimator = "dcdh"
    cells = ()
    _normal_inference = True

    def switchers(self):
        """Switching groups (and their weight) behind each coefficient, by direction."""
        return self._switchers

    def pretrend_test(self):
        """Wald test that the placebos are jointly zero."""
        names = [n for n in self.coefnames if n.startswith("placebo::")]
        if not names:
            raise ValueError("fit with placebos >= 1 to test pre-trends")
        return self.wald_test(keep=[f"^{n}$" for n in names])

    def effects_equal_test(self):
        """Wald test that the effects are all equal (did_multiplegt_dyn's effects_equal)."""
        names = [n for n in self.coefnames if n.startswith("effect::")]
        if len(names) < 2:
            raise ValueError("fit with effects >= 2 to test that they are equal")
        R = np.zeros((len(names) - 1, len(self.coefnames)))
        for j, (a, b) in enumerate(zip(names[:-1], names[1:], strict=True)):
            R[j, self.coefnames.index(a)], R[j, self.coefnames.index(b)] = 1, -1
        return self.wald_test(R=R)

    def _summary_fields(self):
        return {
            **super()._summary_fields(),
            "estimator": PUBLIC,
            "n_units": self.n_units,
            "diagnostics": self.diagnostics,
        }

    def _summary_lines(self):
        return [f"estimator: {PUBLIC}; groups: {self.n_units:,}", *super()._summary_lines()]


def _outcome(fml):
    """The outcome and the covariates' formula terms (None for ``y ~ 1``)."""
    if not isinstance(fml, str) or fml.count("~") != 1:
        raise ValueError("use an outcome formula: 'y ~ 1', or 'y ~ x + z' with covariates")
    y, rhs = [s.strip() for s in fml.split("~")]
    if not y.isidentifier():
        raise ValueError("the outcome must be a column name")
    if "|" in rhs:
        raise ValueError("did_dcdh takes no additional fixed effects: use 'y ~ x + z'")
    return y, (None if rhs == "1" else rhs)


def _count(name, value, low):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < low
    ):
        raise ValueError(f"{name} must be an integer >= {low}")
    return int(value)


class Paths:
    """Each group's baseline, first switch, direction and window, from its treatment path.

    ``D`` is groups by periods, NaN in periods the group is not observed. Periods are
    1..T; F = T + 1 for groups that never switch. As in DIDmultiplegtDYN:

    - the baseline is the treatment in the group's first observed period, and the
      first switch the first observed period with another treatment;
    - groups whose treatment goes both above and below its baseline are dropped, as are
      baselines in which every group first switches in the same period;
    - a group not observed in the period before its switch has an uncertain switch
      date: it is a control until its last observed period before the switch, and its
      later outcomes are not used;
    - each baseline's last period with controls, T_d, is one before the latest period
      any of its groups is known not to have switched by;
    - periods in which no group of a baseline is observed unswitched are not used for it;
    - switchers whose average treatment after the switch (through T_d) equals their
      baseline are dropped.

    ``base`` indexes the retained baselines (-1 for a dropped group) and ``present``
    marks the periods whose outcomes are used.
    """

    def __init__(self, D, T, baselines=None, T_d=None, open_=None):
        n = len(D)
        rows = np.arange(n)
        t = np.arange(1, T + 1)[None, :]
        obs = np.isfinite(D)
        self.observed = obs.any(axis=1)
        first_obs = obs.argmax(axis=1)
        last_obs = T - obs[:, ::-1].argmax(axis=1)  # 1..T
        self.d_sq = D[rows, first_obs]
        dev = np.where(obs, D - self.d_sq[:, None], 0.0)
        self.dev = dev
        self.mixed = ((dev > 0).any(axis=1) & (dev < 0).any(axis=1)) | ~self.observed
        changed = dev != 0
        self.F_first = np.where(changed.any(axis=1), changed.argmax(axis=1) + 1, T + 1)
        self.d_fg = np.where(self.F_first <= T, D[rows, np.minimum(self.F_first, T) - 1], self.d_sq)
        before = obs & (t < self.F_first[:, None])
        self.last_before = np.where(before.any(axis=1), T - before[:, ::-1].argmax(axis=1), 0)
        self.gap = (self.F_first <= T) & (self.last_before < self.F_first - 1)
        self.F = np.where(self.gap, T + 1, self.F_first)
        # the first period the group is not known to be unswitched
        self.F_trunc = np.where(
            self.gap,
            self.last_before + 1,
            np.where(self.F_first > T, last_obs + 1, self.F_first),
        )
        self.unswitched = obs & (t < self.F_first[:, None])
        if baselines is None:
            return
        b = np.searchsorted(baselines, self.d_sq)
        b = np.minimum(b, len(baselines) - 1)
        found = (len(baselines) > 0) & (baselines[b] == self.d_sq) & ~self.mixed
        self.counted = found
        self.Td = np.where(found, T_d[b], 0)
        self.present = (
            obs
            & open_[b[:, None], t]
            & found[:, None]
            & ~(self.gap[:, None] & (t > self.last_before[:, None]))
        )
        window = (t >= self.F[:, None]) & (t <= self.Td[:, None]) & self.present
        filled = window.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            avg = np.where(window, D, 0.0).sum(axis=1) / filled
        back = (filled > 0) & (avg == self.d_sq)
        self.base = np.where(found & ~back, b, -1)
        self.L = self.Td - self.F + 1
        switcher = (self.base >= 0) & (self.F <= T) & (self.L >= 1)
        self.up = np.where(switcher, (avg > self.d_sq).astype(np.int64), -1)

    def outcomes(self, y):
        """The outcomes used: NaN outside ``present``."""
        return np.where(self.present, y, np.nan)


def _dcdh(
    fml,
    data,
    *,
    panel,
    treatment,
    effects=1,
    placebos=0,
    normalized=False,
    switchers="both",
    only_never_switchers=False,
    panel_mode="balanced",
    threads=None,
    memory_gb=None,
    memory_reserve=None,
    workdir=None,
    profile=False,
    batch_units=None,
    preprocess_mb=None,
):
    yname, covariates = _outcome(fml)
    if not isinstance(treatment, str) or not treatment.isidentifier():
        raise ValueError("treatment must name a numeric column")
    if panel is None or len(panel) != 2:
        raise ValueError("give panel=(group, time)")
    effects = _count("effects", effects, 1)
    placebos = _count("placebos", placebos, 0)
    for name, value in (("normalized", normalized), ("only_never_switchers", only_never_switchers)):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be True or False")
    if switchers not in ("both", "in", "out"):
        raise ValueError("switchers must be 'both', 'in' or 'out'")
    if panel_mode not in ("balanced", "complete", "unbalanced"):
        raise ValueError("incomplete_units must be 'error', 'drop' or 'keep'")
    if batch_units is not None:
        _count("batch_units", batch_units, 1)
    if threads is not None:
        _count("n_threads", threads, 1)
    directions = {"both": (1, 0), "in": (1,), "out": (0,)}[switchers]
    if covariates is not None:
        import formulaic

        if treatment in formulaic.Formula(covariates).required_variables:
            raise ValueError("the treatment cannot also be a covariate")

    budget = MemoryBudget(memory_gb, memory_reserve)
    preprocess_mb = preprocess_budget_mb(preprocess_mb, budget)
    baseline_gb = current_rss_gb() or 0
    with thread_scope(threads):
        lf, _, design = inspect(
            f"{yname} ~ 0 + {treatment}" + (f" + {covariates}" if covariates else ""),
            data,
            panel,
            treatment=treatment,
            allow_covariates=True,
            switching=True,
        )
        if not budget.fits((preprocess_mb * 1e6 + 96e6) / 1e9):
            raise MemoryError(
                "preprocessing buffers exceed job headroom before staging; reduce preprocess_mb or request more memory"
            )
        if workdir is not None:
            Path(workdir).mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="diff-stream-", dir=workdir) as run:
            store = PanelStore(
                lf,
                run,
                batch_units or 16384,
                budget,
                preprocess_mb,
                design,
                panel_mode,
                require_never_treated=False,
                switching=True,
            )
            pa.default_memory_pool().release_unused()
            T, N = store.periods, store.n_units
            if design.names[0] != treatment:
                raise ValueError("the treatment cannot also be a covariate")
            covariate_names = design.names[1:]
            Kc = len(covariate_names)
            Lm = min(effects, T - 1)
            P = min(placebos, Lm)
            K = 2 if P else 1

            def buffers(B):
                return 8 * B * (6 * T + 4 * K * Lm) + 1024 * T * T

            B = min(batch_units or 16384, N)
            if batch_units is None:
                while B > 256 and not budget.fits(buffers(B) / 1e9 + 128e6 / 1e9):
                    B //= 2
            store.batch_units = B
            if profile:
                numeric = buffers(B)
                return Profile(
                    model=f"diff-stream {PUBLIC}",
                    threads=threads or detect_cpus(),
                    memory_limit_gb=budget.limit_gb,
                    memory_reserve=budget.reserve,
                    polars_threads=pl.thread_pool_size(),
                    baseline_gb=baseline_gb,
                    components=[
                        Component(
                            "numerical buffers and runtime", numeric + 192e6, numeric * 1.5 + 256e6
                        ),
                        Component(
                            "bounded partition preprocessing",
                            staging_bytes(preprocess_mb, store) + 96e6,
                            preprocess_mb * 1e6 + 256e6,
                        ),
                        Component(
                            "snapshot, hash partitions and sorted leaves",
                            72 * N * T,
                            144 * N * T,
                            where="disk",
                        ),
                    ],
                    notes={
                        "n_units": N,
                        "periods": T,
                        "batch_units": B,
                        "partitions": store.partition_count,
                    },
                    warnings=[
                        "Heuristic estimate: source query, readers, writers and allocator retention can exceed allowances."
                    ],
                )

            def batches():
                for _, y, x, w in store.batches(include_x=True, include_weights=True):
                    yield y, x[:, :, 0], x[:, :, 1:], w

            # Pass 1: the switch periods of each baseline.
            pairs = set()
            switch_values = np.zeros(0)
            reach, opened = {}, {}
            for _, D, _, _ in batches():
                p = Paths(D, T)
                keep = ~p.mixed
                for f in np.unique(p.F_first[keep]).tolist():
                    pairs.update(
                        (d, f) for d in np.unique(p.d_sq[keep & (p.F_first == f)]).tolist()
                    )
                switch_values = np.union1d(switch_values, np.unique(p.d_fg[keep & (p.F <= T)]))
                for d in np.unique(p.d_sq[keep]).tolist():
                    mine = keep & (p.d_sq == d)
                    reach[d] = max(reach.get(d, 0), int(p.F_trunc[mine].max()))
                    seen = np.r_[False, p.unswitched[mine].any(axis=0)]
                    opened[d] = opened[d] | seen if d in opened else seen
                if len({d for d, f in pairs}) > MAX_BASELINES:
                    raise ValueError(
                        f"the first-period treatment takes more than {MAX_BASELINES:,} values; "
                        "groups are compared only with groups of the same first-period treatment"
                    )
            first_switches = {}
            for d, f in pairs:
                first_switches.setdefault(d, set()).add(f)
            valid = sorted(d for d, fs in first_switches.items() if len(fs) >= 2)
            if not valid:
                raise ValueError(
                    "no effect can be estimated: in every first-period treatment, all groups first "
                    "switch in the same period (Design Restriction 1 of de Chaisemartin and "
                    "D'Haultfoeuille)"
                )
            baselines = np.array(valid)
            T_d = np.array([reach[d] - 1 for d in valid], dtype=np.int64)
            open_ = np.stack([opened[d] for d in valid])
            nB = len(baselines)

            # Pass 2: sums by baseline and period, and by switching cohort.
            shape = (K, Lm, nB, T + 1)
            ctrl_w, ctrl_wy, ctrl_n = np.zeros(shape), np.zeros(shape), np.zeros(shape)
            sw_shape = (K, Lm, 2, nB, T + 1)
            sw_w, sw_wy, sw_n = np.zeros(sw_shape), np.zeros(sw_shape), np.zeros(sw_shape)
            sw_cum, sw_now = np.zeros(sw_shape), np.zeros(sw_shape)
            codes, coh = np.zeros(0, dtype=np.int64), np.zeros((0, K, Lm, 3))
            coder = CohortCodes(T, switch_values)
            L_max = {0: 0, 1: 0}
            P_max = {0: 0, 1: 0}
            groups_used = 0
            counted = 0  # groups after the design restriction, as G in DIDmultiplegtDYN
            tgrid = np.arange(1, T + 1)[None, :]
            ctrl_wx = np.zeros(shape + (Kc,))
            sw_wx = np.zeros(sw_shape + (Kc,))
            coh_wx = np.zeros((0, K, Lm, Kc))
            one_n, one_w, one_wy = (
                np.zeros((nB, T + 1)),
                np.zeros((nB, T + 1)),
                np.zeros((nB, T + 1)),
            )
            one_wx, one_wxy = np.zeros((nB, T + 1, Kc)), np.zeros((nB, T + 1, Kc))
            one_wxx = np.zeros((nB, T + 1, Kc, Kc))
            for y, D, X, w in batches():
                p = Paths(D, T, baselines, T_d, open_)
                y = p.outcomes(y)
                X = np.where(p.present[:, :, None], X, np.nan)
                kept = p.base >= 0
                groups_used += int(kept.sum())
                counted += int(p.counted.sum())
                if Kc:
                    # one-period changes of not-yet-switched rows, for the coefficients
                    dY1, dX1 = _changes(y, 1, 0), _changes(X, 1, 0)
                    rows1 = kept[:, None] & (tgrid >= 2) & (tgrid < p.F[:, None])
                    rows1 &= np.isfinite(dY1) & np.isfinite(dX1).all(axis=2)
                    idx1 = (p.base[:, None] * (T + 1) + tgrid)[rows1]
                    w1 = np.broadcast_to(w[:, None], rows1.shape)[rows1]
                    x1, y1 = dX1[rows1], dY1[rows1]
                    one_n += np.bincount(idx1, minlength=nB * (T + 1)).reshape(nB, T + 1)
                    one_w += np.bincount(idx1, w1, nB * (T + 1)).reshape(nB, T + 1)
                    one_wy += np.bincount(idx1, w1 * y1, nB * (T + 1)).reshape(nB, T + 1)
                    for a in range(Kc):
                        one_wx[..., a] += _cells(idx1, w1 * x1[:, a], nB, T)
                        one_wxy[..., a] += _cells(idx1, w1 * x1[:, a] * y1, nB, T)
                        for b in range(Kc):
                            one_wxx[..., a, b] += _cells(idx1, w1 * x1[:, a] * x1[:, b], nB, T)
                for s in (0, 1):
                    mine = p.up == s
                    if mine.any():
                        L_max[s] = max(L_max[s], int(p.L[mine].max()))
                        pl_len = np.where(p.F[mine] >= 3, np.minimum(p.L[mine], p.F[mine] - 2), 0)
                        P_max[s] = max(P_max[s], int(pl_len.max()))
                cell = p.base[:, None] * (T + 1) + tgrid
                unit_code = np.full(len(D), -1, dtype=np.int64)
                sw_all = p.up >= 0
                if sw_all.any():
                    unit_code[sw_all] = coder(
                        p.base[sw_all], p.F[sw_all], p.d_fg[sw_all], p.up[sw_all]
                    )
                    previous = codes
                    codes, coh = _grow(codes, coh, unit_code[sw_all])
                    if Kc:
                        _, coh_wx = _grow(previous, coh_wx, unit_code[sw_all])
                    position = np.searchsorted(codes, unit_code)
                for k in range(K):
                    for li in range(Lm if k == 0 else P):
                        ell = li + 1
                        low = ell + 1 if k == 0 else 2 * ell + 1
                        rows = (
                            kept[:, None]
                            & (tgrid >= low)
                            & (tgrid < p.F[:, None])
                            & (tgrid <= p.Td[:, None])
                        )
                        if only_never_switchers:
                            rows &= (p.F == T + 1)[:, None]
                        change = _changes(y, ell, k)
                        rows &= np.isfinite(change)
                        if k == 1:
                            # a placebo's groups are those of the effect: its change too
                            rows &= np.isfinite(_changes(y, ell, 0))
                        idx = cell[rows]
                        size = nB * (T + 1)
                        ctrl_w[k, li] += np.bincount(idx, (w[:, None] * rows)[rows], size).reshape(
                            nB, T + 1
                        )
                        ctrl_wy[k, li] += np.bincount(
                            idx, (w[:, None] * change)[rows], size
                        ).reshape(nB, T + 1)
                        ctrl_n[k, li] += np.bincount(idx, minlength=size).reshape(nB, T + 1)
                        if Kc:
                            changeX = _changes(X, ell, k)
                            weighted = w[:, None] * rows
                            for a in range(Kc):
                                ctrl_wx[k, li, ..., a] += _cells(
                                    idx, (weighted * changeX[:, :, a])[rows], nB, T
                                )
                        # switchers, at t = F - 1 + l
                        on = (p.up >= 0) & (p.L >= ell)
                        if k == 1:
                            on &= p.F - 1 - ell >= 1
                        rows_on = np.flatnonzero(on)
                        at = p.F[on] - 1 + ell
                        if k == 0:
                            delta = y[rows_on, at - 1] - y[rows_on, p.F[on] - 2]
                        else:
                            delta = y[rows_on, p.F[on] - 2 - ell] - y[rows_on, p.F[on] - 2]
                        # a change needs both of its periods observed; a placebo's
                        # switchers also need the effect's change
                        good = np.isfinite(delta)
                        if k == 1:
                            good &= np.isfinite(y[rows_on, at - 1])
                        on[rows_on[~good]] = False
                        rows_on, at, delta = rows_on[good], at[good], delta[good]
                        if not on.any():
                            continue
                        s = p.up[on]
                        sign = np.where(s == 1, 1.0, -1.0)
                        horizon = (tgrid >= p.F[on][:, None]) & (tgrid <= at[:, None])
                        cum = (p.dev[on] * horizon).sum(axis=1) * sign
                        now = p.dev[rows_on, at - 1] * sign
                        flat = (s * nB + p.base[on]) * (T + 1) + at
                        size2 = 2 * nB * (T + 1)
                        ww = w[on]
                        for target, values in (
                            (sw_w, ww),
                            (sw_wy, ww * delta),
                            (sw_n, np.ones(len(ww))),
                            (sw_cum, ww * cum),
                            (sw_now, ww * now),
                        ):
                            target[k, li] += np.bincount(flat, values, size2).reshape(2, nB, T + 1)
                        where = position[on]
                        for j, values in enumerate((ww, ww * delta, np.ones(len(ww)))):
                            coh[:, k, li, j] += np.bincount(where, values, len(codes))
                        if Kc:
                            if k == 0:
                                deltaX = X[rows_on, at - 1] - X[rows_on, p.F[on] - 2]
                            else:
                                deltaX = X[rows_on, p.F[on] - 2 - ell] - X[rows_on, p.F[on] - 2]
                            for a in range(Kc):
                                sw_wx[k, li, ..., a] += np.bincount(
                                    flat, ww * deltaX[:, a], size2
                                ).reshape(2, nB, T + 1)
                                coh_wx[:, k, li, a] += np.bincount(
                                    where, ww * deltaX[:, a], len(codes)
                                )
            n_eff = min(effects, max(L_max[s] for s in directions))
            if n_eff < 1:
                raise ValueError(
                    "no effect can be estimated: no switcher has a group not yet switched to compare with"
                )
            n_pl = min(placebos, max(P_max[s] for s in directions), n_eff)
            theta = np.zeros((nB, Kc))
            if Kc:
                # Within each baseline, the outcome's one-period changes on the covariates'
                # with period effects, among not-yet-switched rows (did_multiplegt_main).
                with np.errstate(invalid="ignore", divide="ignore"):
                    mean_x = np.where(one_w[..., None] > 0, one_wx / one_w[..., None], 0.0)
                gram_x = one_wxx - np.einsum("btk,btj->btkj", mean_x, one_wx)
                cross_x = one_wxy - mean_x * one_wy[..., None]
                inv_denom = np.zeros((nB, Kc, Kc))
                for b in range(nB):
                    Mb = gram_x[b].sum(axis=0)
                    if np.linalg.cond(Mb) > 1e12:
                        raise ValueError(
                            f"the covariates are collinear, or do not change, among the groups with "
                            f"first-period treatment {baselines[b]:g} that have not switched yet"
                        )
                    invM = np.linalg.inv(Mb)
                    theta[b] = invM @ cross_x[b].sum(axis=0)
                    inv_denom[b] = invM * one_w[b].sum() * counted
                with np.errstate(invalid="ignore", divide="ignore"):
                    fixed = np.where(
                        one_w > 0, (one_wy - np.einsum("btk,bk->bt", one_wx, theta)) / one_w, 0.0
                    )
                ctrl_wy -= np.einsum("klbtj,bj->klbt", ctrl_wx, theta)
                sw_wy -= np.einsum("klsbtj,bj->klsbt", sw_wx, theta)
                if len(codes):
                    cb0 = coder.parts(codes)[0]
                    coh[:, :, :, 1] -= np.einsum("cklj,cj->ckl", coh_wx, theta[cb0])

            # Cells with controls; switchers in cells without any are not used.
            valid_cell = ctrl_w > 0
            if K == 2:
                valid_cell[1] &= ctrl_w[0] > 0
            sw_wx *= valid_cell[:, :, None, :, :, None]
            for s in (0, 1):
                if s not in directions:
                    sw_wx[:, :, s] = 0
            for arr in (sw_w, sw_wy, sw_n, sw_cum, sw_now):
                arr *= valid_cell[:, :, None]
                for s in (0, 1):
                    if s not in directions:
                        arr[:, :, s] = 0
            with np.errstate(invalid="ignore", divide="ignore"):
                ctrl_mean = np.where(ctrl_w > 0, ctrl_wy / ctrl_w, 0.0)
                t_any = sw_w.sum(axis=3) > 0
                gate = t_any[:, :, :, None, :]
                un_w = sw_w + ctrl_w[:, :, None] * gate
                union_n = sw_n + ctrl_n[:, :, None] * gate
                union_mean = np.where(un_w > 0, (sw_wy + ctrl_wy[:, :, None] * gate) / un_w, 0.0)
            # cohort tables, with cohorts in cells without controls removed
            cb, cF, cs = coder.parts(codes)
            cohort_n = coh[:, :, :, 2].transpose(1, 2, 0).copy()
            cohort_w = coh[:, :, :, 0].transpose(1, 2, 0)
            cohort_wy = coh[:, :, :, 1].transpose(1, 2, 0)
            for k in range(K):
                for li in range(Lm):
                    at = np.minimum(cF - 1 + li + 1, T)
                    ok = (
                        valid_cell[k, li, cb, at] & (cF - 1 + li + 1 <= T) & np.isin(cs, directions)
                    )
                    cohort_n[k, li] *= ok
            with np.errstate(invalid="ignore", divide="ignore"):
                cohort_mean = np.where(cohort_w > 0, cohort_wy / cohort_w, 0.0)

            # Pass 3: each group's terms.
            Nk = sw_w.sum(axis=(3, 4))  # (K, Lm, 2)
            columns = [(0, li) for li in range(n_eff)] + [(1, li) for li in range(n_pl)]
            width = len(columns) + 1
            coef = np.zeros((width, K, Lm, 2))
            for j, (k, li) in enumerate(columns):
                total = Nk[k, li].sum()
                if total > 0:
                    coef[j, k, li, 1], coef[j, k, li, 0] = 1 / total, -1 / total
                    if normalized:
                        delta = sw_cum[k, li].sum() / total
                        coef[j] /= delta
            # the average total effect: within each direction, the effects weighted by
            # their switchers over the average treatment change; then the directions
            # weighted by their switchers times that change (did_multiplegt_dyn)
            sums = {s: Nk[0, :n_eff, s].sum() for s in (0, 1)}
            den = {s: sw_now[0, :n_eff, s].sum() / sums[s] if sums[s] else 0.0 for s in (0, 1)}
            weight_in = den[1] * sums[1]
            weight_all = den[1] * sums[1] + den[0] * sums[0]
            share = {1: weight_in / weight_all if weight_all > 0 else 0.5}
            share[0] = 1 - share[1]
            for s, sign in ((1, 1.0), (0, -1.0)):
                if sums[s] and den[s]:
                    coef[-1, 0, :n_eff, s] = share[s] * sign / (sums[s] * den[s])
            if Kc:
                # M of did_multiplegt_dyn_core: the switchers' covariate changes less their
                # controls', per switcher; used where l <= T_d - 2
                with np.errstate(invalid="ignore", divide="ignore"):
                    ratio = np.where(ctrl_w[:, :, None] > 0, sw_w / ctrl_w[:, :, None], 0.0)
                    moved = sw_wx.sum(axis=4) - np.einsum("klsbt,klbtj->klsbj", ratio, ctrl_wx)
                    Mc = np.where(Nk[..., None, None] > 0, moved / Nk[..., None, None], 0.0)
                horizon = np.arange(1, Lm + 1)[None, :, None, None]
                Mc *= (horizon <= T_d[None, None, None, :] - 2)[..., None]
                constant = np.einsum("klsbj,bj->kls", Mc, theta)
            estimate = np.zeros(width)
            gram = np.zeros((width, width))
            arrays = (
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
            )
            for y, D, X, w in batches():
                p = Paths(D, T, baselines, T_d, open_)
                y = p.outcomes(y)
                X = np.where(p.present[:, :, None], X, np.nan)
                kept = p.base >= 0
                if Kc:
                    th = np.where(kept[:, None], theta[np.maximum(p.base, 0)], 0.0)
                    y_raw, y = y, y - np.einsum("btk,bk->bt", X, th)
                cid = np.full(len(D), -1, dtype=np.int64)
                sw = p.up >= 0
                if sw.any() and len(codes):
                    code = coder(p.base[sw], p.F[sw], p.d_fg[sw], p.up[sw])
                    pos = np.minimum(np.searchsorted(codes, code), len(codes) - 1)
                    cid[sw] = np.where(codes[pos] == code, pos, -1)
                U, V = group_terms(
                    p.base, p.F, p.up, cid, w, y, T_d, only_never_switchers, T, *arrays
                )
                if Kc:
                    b0 = np.maximum(p.base, 0)
                    dY1, dX1 = _changes(y_raw, 1, 0), _changes(X, 1, 0)
                    rows1 = kept[:, None] & (tgrid >= 2) & (tgrid < p.F[:, None])
                    rows1 &= np.isfinite(dY1) & np.isfinite(dX1).all(axis=2)
                    n1 = one_n[b0[:, None], tgrid]
                    factor = np.where(n1 >= 2, np.sqrt(n1 / np.maximum(n1 - 1, 1)), 1.0)
                    E = np.where(
                        n1 >= 2, np.einsum("btk,bk->bt", dX1, th) + fixed[b0[:, None], tgrid], 0.0
                    )
                    N_c = one_w[b0].sum(axis=1)[:, None]
                    term = np.where(rows1, w[:, None] * factor * (dY1 - E) / N_c, 0.0)
                    resid = np.nan_to_num(dX1 - mean_x[b0[:, None], tgrid])
                    in_sum = np.einsum("btk,bt->bk", resid, term)
                    own = np.einsum("bkj,bj->bk", inv_denom[b0], in_sum) * (p.F >= 3)[:, None]
                    by_group = Mc[:, :, :, b0, :].transpose(3, 0, 1, 2, 4)
                    part2 = np.einsum("bklsj,bj->bkls", by_group, own)
                    part2 = part2 - constant[None]
                    V -= np.where(kept[:, None, None, None], Nk[None] * part2 / counted, 0.0)
                estimate += np.einsum("bkls,jkls->j", U, coef)
                scores = np.einsum("bkls,jkls->bj", V, coef)
                gram += scores.T @ scores

            names = (
                [f"effect::{li + 1}" for li in range(n_eff)]
                + [f"placebo::{li + 1}" for li in range(n_pl)]
                + ["average_total_effect"]
            )
            defined = np.array([Nk[k, li].sum() > 0 for k, li in columns] + [weight_all > 0])
            estimate[~defined] = np.nan
            gram[~defined, :] = np.nan
            gram[:, ~defined] = np.nan
            table = pl.DataFrame(
                {
                    "coefficient": names[:-1],
                    "switchers_in": [int(sw_n[k, li, 1].sum()) for k, li in columns],
                    "switchers_out": [int(sw_n[k, li, 0].sum()) for k, li in columns],
                    "weight_in": [float(sw_w[k, li, 1].sum()) for k, li in columns],
                    "weight_out": [float(sw_w[k, li, 0].sum()) for k, li in columns],
                }
            )
            labels = store.period_labels
            diagnostics = dict(
                switchers=switchers,
                covariates=covariate_names,
                only_never_switchers=only_never_switchers,
                normalized=normalized,
                groups_input=store.n_units_input,
                groups_used=groups_used,
                periods=T,
                time_min=labels.label_int(int(store.times[0])),
                time_max=labels.label_int(int(store.times[-1])),
                baselines=baselines.tolist(),
                last_period_with_controls={
                    str(d): labels.label_int(int(store.times[t - 1]))
                    for d, t in zip(baselines.tolist(), T_d.tolist(), strict=True)
                },
                inference="clustered by group; the authors' conservative variance (cohort-demeaned)",
                batch_units=B,
                partitions=store.partition_count,
            )
            return DCDHResult(
                fml=fml,
                depvar=yname,
                coefnames=names,
                beta=estimate,
                vcov=gram,
                vcov_type="CRV0",
                df_t=N - 1,
                all_vcovs={"CRV0": (gram, N - 1)},
                n_obs=store.n_rows_used,
                n_clusters={panel[0]: N},
                n_units=N,
                diagnostics=diagnostics,
                _switchers=table,
                _internal=dict(diagnostics),
            )


def _cells(index, values, nB, T):
    """Sums of ``values`` by baseline-period cell (``index`` = b * (T + 1) + t)."""
    return np.bincount(index, values, nB * (T + 1)).reshape(nB, T + 1)


def _changes(y, ell, k):
    """Each row's change for horizon l: y_t - y_{t-l} (effect) or y_{t-2l} - y_{t-l}
    (placebo); NaN where a period is missing or out of range."""
    out = np.full(y.shape, np.nan)
    if k == 0:
        out[:, ell:] = y[:, ell:] - y[:, :-ell]
    elif 2 * ell < y.shape[1]:
        out[:, 2 * ell :] = y[:, : -2 * ell] - y[:, ell:-ell]
    return out


class CohortCodes:
    """One integer per switching cohort: (baseline, first switch, treatment at the
    switch, direction), the cohorts of the authors' variance."""

    def __init__(self, T, switch_values):
        self.T, self.values = T, np.asarray(switch_values, dtype=float)
        self.n = max(len(self.values), 1)

    def __call__(self, base, F, d_fg, up):
        value = np.searchsorted(self.values, d_fg)
        return ((base * (self.T + 2) + F) * self.n + value) * 2 + up

    def parts(self, codes):
        rest = codes // 2
        timing = rest // self.n
        return timing // (self.T + 2), timing % (self.T + 2), codes % 2


def _grow(codes, coh, new):
    """Add the cohorts in ``new`` that ``codes`` (sorted) does not have yet."""
    fresh = np.setdiff1d(new, codes, assume_unique=False)
    if not len(fresh):
        return codes, coh
    merged = np.union1d(codes, fresh)
    out = np.zeros((len(merged),) + coh.shape[1:])
    out[np.searchsorted(merged, codes)] = coh
    return merged, out


@_docs.document
def did_dcdh(
    fml,
    data,
    *,
    panel,
    treatment,
    effects=1,
    placebos=0,
    normalized=False,
    switchers="both",
    only_never_switchers=False,
    incomplete_units="error",
    n_threads=None,
    max_memory_gb=None,
    memory_reserve=None,
    workdir=None,
    profile=False,
    batch_units=None,
    preprocess_mb=None,
):
    """de Chaisemartin and D'Haultfoeuille's event-study estimator for a treatment that may
    switch on and off and take several values.

    Each group's baseline is its first-period treatment, and its first switch the first
    period its treatment differs from that. Effect l compares the outcome change of the
    groups that first switched l - 1 periods earlier, since the period before their
    switch, with that of groups with the same baseline that have not switched yet. As
    the authors' R and Stata ``did_multiplegt_dyn``.

    de Chaisemartin, C. and D'Haultfoeuille, X. (2024). Difference-in-differences
    estimators of intertemporal treatment effects. *Review of Economics and Statistics*,
    forthcoming.

    Parameters
    ----------
    fml : str
        ``"y ~ 1"``. Covariates are not supported yet.
    data : polars.DataFrame, polars.LazyFrame, str or Path
        The panel, or a Parquet path or glob, read in streaming batches.
    panel : tuple of str
        The group and time columns, ``(group, time)``; periods are evenly spaced
        positive integers.
    treatment : str
        The treatment, numeric: binary or with several values, and free to change
        in any period. Groups are compared only with groups of the same first-period
        treatment.
    effects : int, default 1
        The number of effects, l = 1, ..., effects (fewer if the data support fewer).
    placebos : int, default 0
        The number of placebos (at most ``effects``).
    normalized : bool, default False
        Divide each effect by the average change in treatment the switchers have
        received up to it, giving effects per unit of treatment.
    switchers : {"both", "in", "out"}, default "both"
        Use switchers whose treatment rose ("in"), fell ("out"), or both.
    only_never_switchers : bool, default False
        Compare switchers only with groups whose treatment never changes.
    incomplete_units : {"error", "drop", "keep"}, default "error"
        What to do with groups missing some periods: refuse the data, drop those
        groups, or keep their observed periods. With "keep", a change is used only where
        both of its periods are observed, and a group not observed just before its
        switch is used only as a control until its last observed period before it, as
        in DIDmultiplegtDYN.
    {RESOURCES}

    Returns
    -------
    DCDHResult or Profile
        ``effect::l``, ``placebo::l`` and ``average_total_effect`` with their joint
        covariance; ``pretrend_test()``, ``effects_equal_test()``, ``switchers()``.
    """
    options = dict(locals())
    fml, data = options.pop("fml"), options.pop("data")
    return public_profile(_dcdh(fml, data, **_options.internal(options)))
