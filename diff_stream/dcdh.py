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

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
import pyarrow as pa
from hdfe_stream.core.partition import preprocess_budget_mb
from hdfe_stream.core.profile import Component, Profile
from hdfe_stream.core.results_base import RegressionResult
from stream_limits import MemoryBudget, current_rss_gb, detect_cpus, thread_scope

from . import _docs, _options
from ._core.dcdh_kernels import group_terms
from ._core.partition import staging_bytes
from ._present import public_profile
from .continuous_covariates import join_levels
from .store import PanelStore, inspect, source

PUBLIC = "did_dcdh"


class ContinuousBaselineWarning(UserWarning):
    """did_dcdh's standard errors with a continuous first-period treatment may be too
    small, as the authors note for their continuous option."""


# Distinct first-period treatment values (within trend groups): each needs a row of
# every cell table.
MAX_BASELINES = 10_000
# The column that carries each row's trend group (trends_by) through staging.
TREND = "__dcdh_trend"
# The column that carries each row's cluster (cluster=) through staging.
CLUSTER = "__dcdh_cluster"


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
    """Each group's cell, first switch, direction and window, from its treatment path.

    ``D`` is groups by periods, NaN in periods the group is not observed, and ``code``
    the group's trend group (0 without ``trends_by``). Periods are 1..T; F = T + 1 for
    groups that never switch. A cell is a first-period treatment within a trend group,
    keyed ``d + 1j * code``. As in DIDmultiplegtDYN:

    - the baseline is the treatment in the group's first observed period, and the
      first switch the first observed period with another treatment;
    - groups whose treatment goes both above and below its baseline are dropped, as are
      cells in which every group first switches in the same period (there is no one
      to compare with);
    - a group not observed in the period before its switch has an uncertain switch
      date: it is a control until its last observed period before the switch, and its
      later outcomes are not used;
    - with linear trends, groups first switching in period 2 are dropped (they have no
      trend to remove), after the design restriction;
    - each cell's last period with controls, T_d, is one before the latest period any
      of its groups is known not to have switched by;
    - periods in which no group of a cell is observed unswitched are not used for it;
    - switchers whose average treatment after the switch (through T_d) equals their
      baseline are dropped.

    ``base`` indexes the retained cells (-1 for a dropped group) and ``present`` marks
    the periods whose outcomes are used.
    """

    def __init__(
        self,
        D,
        T,
        code=None,
        cells=None,
        T_d=None,
        open_=None,
        linear_trends=False,
        continuous=False,
    ):
        n = len(D)
        rows = np.arange(n)
        t = np.arange(1, T + 1)[None, :]
        obs = np.isfinite(D)
        self.observed = obs.any(axis=1)
        first_obs = obs.argmax(axis=1)
        last_obs = T - obs[:, ::-1].argmax(axis=1)  # 1..T
        self.d_sq = D[rows, first_obs]
        self.code = np.zeros(n) if code is None else code
        # with a continuous first-period treatment, all groups share one cell (within
        # trend groups), as DIDmultiplegtDYN's continuous option
        self.key = (0.0 if continuous else self.d_sq) + 1j * self.code
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
        # dropped with linear trends, though counted in G and in the design restriction
        self.second = linear_trends & (self.F == 2)
        # the first period the group is not known to be unswitched
        self.F_trunc = np.where(
            self.gap,
            self.last_before + 1,
            np.where(self.F_first > T, last_obs + 1, self.F_first),
        )
        self.unswitched = obs & (t < self.F_first[:, None])
        if cells is None:
            return
        b = np.minimum(np.searchsorted(cells, self.key), len(cells) - 1)
        found = (len(cells) > 0) & (cells[b] == self.key) & ~self.mixed
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
        self.base = np.where(found & ~back & ~self.second, b, -1)
        self.L = self.Td - self.F + 1
        switcher = (self.base >= 0) & (self.F <= T) & (self.L >= 1)
        self.up = np.where(switcher, (avg > self.d_sq).astype(np.int64), -1)
        # the treatment at the switch that, with the cell and the switch period, makes
        # a switching cohort of the authors' variance; with a continuous first-period
        # treatment, the switch's direction (DIDmultiplegtDYN's binarized treatment)
        self.switch_value = np.where(self.up == 1, 1.0, -1.0) if continuous else self.d_fg

    def outcomes(self, y):
        """The outcomes used: NaN outside ``present``."""
        return np.where(self.present, y, np.nan)


def _trend_groups(trends_by, src, reserved):
    """The trend-group columns, their distinct values (one code each) and the source
    with the code joined on as ``TREND``."""
    columns = [trends_by] if isinstance(trends_by, str) else list(trends_by)
    if not columns or not all(isinstance(c, str) and c for c in columns):
        raise ValueError("trends_by must be a column name or a list of column names")
    if len(set(columns)) != len(columns):
        raise ValueError("trends_by lists a column twice")
    clash = set(columns) & set(reserved)
    if clash:
        raise ValueError(
            f"trends_by cannot include the outcome, panel, treatment or weights: {sorted(clash)}"
        )
    missing = set(columns) - set(src.collect_schema())
    if missing:
        raise ValueError(f"unknown trends_by columns: {sorted(missing)}")
    levels = src.select(columns).unique().collect(engine="streaming")
    if levels.null_count().sum_horizontal().item():
        raise ValueError("trends_by has missing values; fill or drop them first")
    if len(levels) > MAX_BASELINES:
        raise ValueError(f"trends_by takes more than {MAX_BASELINES:,} distinct values")
    levels = levels.sort(columns).with_row_index(TREND)
    labels = [", ".join(str(v) for v in row) for row in levels.drop(TREND).iter_rows()]
    joined = src.join(
        levels.with_columns(pl.col(TREND).cast(pl.Float64)).lazy(), on=columns, how="left"
    )
    return columns, labels, joined


def _group_codes(codes):
    """Each group's trend-group code, which must not change over its observed periods."""
    high, low = np.fmax.reduce(codes, axis=1), np.fmin.reduce(codes, axis=1)
    if np.any((high != low) & np.isfinite(high)):
        raise ValueError("trends_by must be constant within each group")
    return np.nan_to_num(high)


def _slots(Lm, P, linear_trends):
    """The horizons estimated: (kind, horizon index, same-switcher horizon r).

    Without linear trends, effect l and placebo l are each one horizon of all their
    switchers (r = 0). With linear trends, effect (placebo) l sums horizons 1..l, each
    for the switchers that have all l: r = l."""
    if not linear_trends:
        return [(0, li, 0) for li in range(Lm)] + [(1, li, 0) for li in range(P)]
    return [(0, k - 1, i) for i in range(1, Lm + 1) for k in range(1, i + 1)] + [
        (1, k - 1, i) for i in range(1, P + 1) for k in range(1, i + 1)
    ]


def _first_difference(a):
    """a_t - a_{t-1} along periods (axis 1); NaN unless both periods are observed."""
    out = np.full(a.shape, np.nan)
    out[:, 1:] = a[:, 1:] - a[:, :-1]
    return out


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
    trends_by=None,
    linear_trends=False,
    weights=None,
    allow_time_varying_weights=False,
    panel_mode="balanced",
    cluster=None,
    continuous_baseline=None,
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
    for name, value in (
        ("normalized", normalized),
        ("only_never_switchers", only_never_switchers),
        ("linear_trends", linear_trends),
        ("allow_time_varying_weights", allow_time_varying_weights),
    ):
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
    if cluster is not None and (not isinstance(cluster, str) or not cluster):
        raise ValueError("cluster must name a column")
    if cluster == panel[0]:
        cluster = None  # the group is the default cluster, as in DIDmultiplegtDYN
    degree = (
        None
        if continuous_baseline is None
        else _count("continuous_baseline", continuous_baseline, 1)
    )
    continuous = degree is not None
    if covariates is not None:
        import formulaic

        if treatment in formulaic.Formula(covariates).required_variables:
            raise ValueError("the treatment cannot also be a covariate")

    budget = MemoryBudget(memory_gb, memory_reserve)
    preprocess_mb = preprocess_budget_mb(preprocess_mb, budget)
    baseline_gb = current_rss_gb() or 0
    with thread_scope(threads):
        src = source(data)
        trend_columns, trend_labels = [], ["all"]
        if trends_by is not None:
            reserved = [yname, *panel, treatment] + ([weights] if isinstance(weights, str) else [])
            trend_columns, trend_labels, src = _trend_groups(trends_by, src, reserved)
        n_trend = 1 if trends_by is not None else 0
        n_clusters = 0
        if cluster is not None:
            if cluster in (yname, panel[1], treatment):
                raise ValueError("cluster must be a column above the group")
            src, n_clusters = join_levels(src, [cluster], CLUSTER, "cluster")
            if n_clusters < 2:
                raise ValueError("cluster needs at least two clusters")
        n_cl = 1 if cluster is not None else 0
        rhs = " + ".join(
            [treatment]
            + ([TREND] if n_trend else [])
            + ([CLUSTER] if n_cl else [])
            + ([covariates] if covariates else [])
        )
        lf, _, design = inspect(
            f"{yname} ~ 0 + {rhs}",
            src,
            panel,
            treatment=treatment,
            allow_covariates=True,
            weights=weights,
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
                allow_time_varying_weights=allow_time_varying_weights,
                switching=True,
            )
            pa.default_memory_pool().release_unused()
            T, N = store.periods, store.n_units
            if design.names[0] != treatment:
                raise ValueError("the treatment cannot also be a covariate")
            covariate_names = design.names[1 + n_trend + n_cl :]
            if any(name in trend_columns for name in covariate_names):
                raise ValueError("a trends_by column cannot also be a covariate")
            # with a continuous first-period treatment, its powers interacted with
            # being at or after each period (from the second) are added as covariates,
            # so that untreated trends are a polynomial in it, period by period
            Kc = len(covariate_names) + (T - 1) * (degree or 0)
            Lm = min(effects, T - 1)
            P = min(placebos, Lm)
            K = 2 if P else 1
            slots = _slots(Lm, P, linear_trends)
            S = len(slots)
            slot_kind = np.array([k for k, _, _ in slots], dtype=np.int64)
            slot_horizon = np.array([li for _, li, _ in slots], dtype=np.int64)
            slot_index = {slot: j for j, slot in enumerate(slots)}

            def buffers(B):
                numeric = 8 * B * (8 * T + 4 * S + 2 * S * Kc) + B * S + 1024 * T * T
                if n_cl:
                    # the clusters' score sums, and the (cell-period or cohort, cluster)
                    # pairs kept for the demeaning's cluster counts
                    numeric += 8 * n_clusters * (2 * S + 2)
                    numeric += 32 * S * T * min(N, 64 * n_clusters)
                return numeric

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
                            80 * N * T,
                            160 * N * T,
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
                for _, y, x, w in store.batches(
                    include_x=True, include_weights=True, row_weights=True
                ):
                    code = _group_codes(x[:, :, 1]) if n_trend else None
                    cl = _group_cluster(x[:, :, 1 + n_trend]) if n_cl else None
                    X = x[:, :, 1 + n_trend + n_cl :]
                    if continuous:
                        X = np.concatenate([X, _baseline_polynomial(x[:, :, 0], degree)], axis=2)
                    if linear_trends:
                        y, X = _first_difference(y), _first_difference(X)
                    yield y, x[:, :, 0], X, w, code, cl

            # Pass 1: the switch periods of each cell.
            pairs = set()
            switch_values = np.zeros(0)
            reach, opened = {}, {}
            for _, D, _, _, code, _ in batches():
                p = Paths(D, T, code, linear_trends=linear_trends, continuous=continuous)
                keep = ~p.mixed
                for f in np.unique(p.F_first[keep]).tolist():
                    pairs.update((c, f) for c in np.unique(p.key[keep & (p.F_first == f)]).tolist())
                if continuous:
                    switch_values = np.array([-1.0, 1.0])
                else:
                    switch_values = np.union1d(switch_values, np.unique(p.d_fg[keep & (p.F <= T)]))
                for c in np.unique(p.key[keep]).tolist():
                    mine = keep & (p.key == c)
                    used = mine & ~p.second
                    if used.any():
                        reach[c] = max(reach.get(c, 0), int(p.F_trunc[used].max()))
                    seen = np.r_[False, p.unswitched[mine].any(axis=0)]
                    opened[c] = opened[c] | seen if c in opened else seen
                if len({c for c, f in pairs}) > MAX_BASELINES:
                    raise ValueError(
                        f"the first-period treatment takes more than {MAX_BASELINES:,} values "
                        "(within trend groups); groups are compared only with groups of the "
                        "same first-period treatment"
                    )
            first_switches = {}
            for c, f in pairs:
                first_switches.setdefault(c, set()).add(f)
            valid = np.sort(np.array([c for c, fs in first_switches.items() if len(fs) >= 2]))
            valid = valid.tolist()
            if not valid:
                raise ValueError(
                    "no effect can be estimated: in every first-period treatment, all groups first "
                    "switch in the same period (Design Restriction 1 of de Chaisemartin and "
                    "D'Haultfoeuille)"
                )
            cells = np.array(valid, dtype=complex)
            T_d = np.array([reach.get(c, 1) - 1 for c in valid], dtype=np.int64)
            open_ = np.stack([opened[c] for c in valid])
            nB = len(cells)
            # baselines: the first-period treatments; theta is estimated per baseline,
            # pooled over its trend groups
            baselines, level = np.unique(cells.real, return_inverse=True)
            level = level.ravel()
            nL = len(baselines)

            # Pass 2: sums by cell and period for the controls, and by cell, period and
            # switching cohort for the switchers (with linear trends, in a pass of their
            # own: which switchers each estimate uses depends on the controls' sums).
            shape = (K, Lm, nB, T + 1)
            ctrl_w, ctrl_wy, ctrl_n = np.zeros(shape), np.zeros(shape), np.zeros(shape)
            ctrl_wx = np.zeros(shape + (Kc,))
            lead_w = np.zeros((P, nB, T + 1))  # controls for the placebos' same-switcher check
            sw_shape = (S, 2, nB, T + 1)
            sw_w, sw_wy, sw_n = np.zeros(sw_shape), np.zeros(sw_shape), np.zeros(sw_shape)
            sw_cum, sw_now = np.zeros(sw_shape), np.zeros(sw_shape)
            sw_wx = np.zeros(sw_shape + (Kc,))
            coder = CohortCodes(T, switch_values)
            cohorts = dict(codes=np.zeros(0, dtype=np.int64), coh=np.zeros((0, S, 3)))
            cohorts["wx"] = np.zeros((0, S, Kc))
            L_max = {0: 0, 1: 0}
            P_max = {0: 0, 1: 0}
            groups_used = 0
            counted = 0  # groups after the design restriction, as G in DIDmultiplegtDYN
            tgrid = np.arange(1, T + 1)[None, :]
            one_n, one_w, one_wy = (np.zeros((nB, T + 1)) for _ in range(3))
            one_wx, one_wxy = np.zeros((nB, T + 1, Kc)), np.zeros((nB, T + 1, Kc))
            one_wxx = np.zeros((nB, T + 1, Kc, Kc))
            horizons = np.arange(1, Lm + 1)

            def eligibility(p, y):
                """Whether each group may be a switcher in each slot: with linear trends,
                only if it has controls and observed changes at every horizon up to the
                slot's r (and, for placebos, observed placebo changes up to r)."""
                E = np.ones((len(y), S), dtype=bool)
                if not linear_trends:
                    return E
                rows = np.flatnonzero(p.up >= 0)
                Fr, br = p.F[rows], p.base[rows]
                last = y[rows, Fr - 2]
                ok = np.zeros((len(rows), Lm), dtype=bool)
                for q in range(1, Lm + 1):
                    at = np.minimum(Fr - 1 + q, T)
                    ok[:, q - 1] = (
                        (Fr - 1 + q <= T)
                        & (ctrl_w[0, q - 1, br, at] > 0)
                        & np.isfinite(y[rows, at - 1] - last)
                    )
                still = np.logical_and.accumulate(ok, axis=1) & (
                    Fr[:, None] - 1 + horizons[None, :] <= p.Td[rows, None]
                )
                fill = np.zeros((len(rows), P), dtype=bool)
                for q in range(1, P + 1):
                    at = np.maximum(Fr - 1 - q, 1)
                    fill[:, q - 1] = (
                        (Fr - 1 - q >= 1)
                        & (lead_w[q - 1, br, at] > 0)
                        & np.isfinite(y[rows, at - 1] - last)
                    )
                fill = np.logical_and.accumulate(fill, axis=1)
                for j, (k, _, r) in enumerate(slots):
                    E[rows, j] = still[:, r - 1] & (fill[:, r - 1] if k == 1 else True)
                return E

            # With clusters, the authors' cohort demeaning counts distinct clusters, not
            # groups, in each set of controls, switching cohort, and their union: the
            # (cell-period or cohort, cluster) pairs seen are kept for those counts.
            ctrl_pairs, sw_pairs, coh_pairs = {}, {}, {}

            def remember(store_, key, values):
                if len(values):
                    old = store_.get(key)
                    store_[key] = np.unique(values if old is None else np.r_[old, values], axis=0)

            def add_controls(p, y, X, w, cl=None):
                nonlocal groups_used, counted
                kept = p.base >= 0
                groups_used += int(kept.sum())
                counted += int(p.counted.sum())
                size = nB * (T + 1)
                if Kc:
                    # one-period changes of not-yet-switched rows, for the coefficients
                    dY1, dX1 = _changes(y, 1, 0), _changes(X, 1, 0)
                    rows1 = kept[:, None] & (tgrid >= 2) & (tgrid < p.F[:, None])
                    rows1 &= np.isfinite(dY1) & np.isfinite(dX1).all(axis=2)
                    idx1 = (p.base[:, None] * (T + 1) + tgrid)[rows1]
                    w1, x1, y1 = w[rows1], dX1[rows1], dY1[rows1]
                    one_n[:] += np.bincount(idx1, minlength=size).reshape(nB, T + 1)
                    one_w[:] += _cells(idx1, w1, nB, T)
                    one_wy[:] += _cells(idx1, w1 * y1, nB, T)
                    for a in range(Kc):
                        one_wx[..., a] += _cells(idx1, w1 * x1[:, a], nB, T)
                        one_wxy[..., a] += _cells(idx1, w1 * x1[:, a] * y1, nB, T)
                        for c in range(Kc):
                            one_wxx[..., a, c] += _cells(idx1, w1 * x1[:, a] * x1[:, c], nB, T)
                for s in (0, 1):
                    mine = p.up == s
                    if mine.any():
                        L_max[s] = max(L_max[s], int(p.L[mine].max()))
                        pl_len = np.where(p.F[mine] >= 3, np.minimum(p.L[mine], p.F[mine] - 2), 0)
                        P_max[s] = max(P_max[s], int(pl_len.max()))
                cell = p.base[:, None] * (T + 1) + tgrid
                eligible_ctrl = kept[:, None] & (tgrid < p.F[:, None])
                if only_never_switchers:
                    eligible_ctrl &= (p.F == T + 1)[:, None]
                for k in range(K):
                    for li in range(Lm if k == 0 else P):
                        ell = li + 1
                        low = ell + 1 if k == 0 else 2 * ell + 1
                        rows = eligible_ctrl & (tgrid >= low) & (tgrid <= p.Td[:, None])
                        change = _changes(y, ell, k)
                        rows &= np.isfinite(change)
                        if k == 1:
                            # a placebo's groups are those of the effect: its change too
                            rows &= np.isfinite(_changes(y, ell, 0))
                        idx = cell[rows]
                        if cl is not None:
                            cl_rows = np.broadcast_to(cl[:, None], rows.shape)[rows]
                            remember(ctrl_pairs, (k, li), idx * n_clusters + cl_rows)
                        ctrl_w[k, li] += _cells(idx, w[rows], nB, T)
                        ctrl_wy[k, li] += _cells(idx, (w * change)[rows], nB, T)
                        ctrl_n[k, li] += np.bincount(idx, minlength=size).reshape(nB, T + 1)
                        if Kc:
                            changeX = _changes(X, ell, k)
                            for a in range(Kc):
                                ctrl_wx[k, li, ..., a] += _cells(
                                    idx, (w * changeX[:, :, a])[rows], nB, T
                                )
                if linear_trends:
                    for q in range(1, P + 1):
                        lead = np.full(y.shape, np.nan)
                        lead[:, :-q] = y[:, :-q] - y[:, q:]
                        rows = eligible_ctrl & np.isfinite(lead)
                        lead_w[q - 1] += _cells(cell[rows], w[rows], nB, T)

            def add_switchers(p, y, X, w, eligible, cl=None):
                unit_code = np.full(len(y), -1, dtype=np.int64)
                sw_all = p.up >= 0
                if not sw_all.any():
                    return
                unit_code[sw_all] = coder(
                    p.base[sw_all], p.F[sw_all], p.switch_value[sw_all], p.up[sw_all]
                )
                previous = cohorts["codes"]
                cohorts["codes"], cohorts["coh"] = _grow(
                    previous, cohorts["coh"], unit_code[sw_all]
                )
                if Kc:
                    _, cohorts["wx"] = _grow(previous, cohorts["wx"], unit_code[sw_all])
                codes = cohorts["codes"]
                position = np.searchsorted(codes, unit_code)
                size2 = 2 * nB * (T + 1)
                for j, (k, li, _) in enumerate(slots):
                    ell = li + 1
                    on = (p.up >= 0) & (p.L >= ell) & eligible[:, j]
                    if k == 1:
                        on &= p.F - 1 - ell >= 1
                    rows_on = np.flatnonzero(on)
                    at = p.F[on] - 1 + ell
                    if k == 0:
                        delta = y[rows_on, at - 1] - y[rows_on, p.F[on] - 2]
                    else:
                        delta = y[rows_on, p.F[on] - 2 - ell] - y[rows_on, p.F[on] - 2]
                    # a change needs both of its periods observed; a placebo's switchers
                    # also need the effect's change
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
                    ww = w[rows_on, at - 1]
                    if cl is not None:
                        remember(sw_pairs, j, flat * n_clusters + cl[rows_on])
                        remember(coh_pairs, j, np.column_stack([unit_code[rows_on], cl[rows_on]]))
                    for target, values in (
                        (sw_w, ww),
                        (sw_wy, ww * delta),
                        (sw_n, np.ones(len(ww))),
                        (sw_cum, ww * cum),
                        (sw_now, ww * now),
                    ):
                        target[j] += np.bincount(flat, values, size2).reshape(2, nB, T + 1)
                    where = position[on]
                    for c, values in enumerate((ww, ww * delta, np.ones(len(ww)))):
                        cohorts["coh"][:, j, c] += np.bincount(where, values, len(codes))
                    if Kc:
                        if k == 0:
                            deltaX = X[rows_on, at - 1] - X[rows_on, p.F[on] - 2]
                        else:
                            deltaX = X[rows_on, p.F[on] - 2 - ell] - X[rows_on, p.F[on] - 2]
                        for a in range(Kc):
                            sw_wx[j, ..., a] += np.bincount(flat, ww * deltaX[:, a], size2).reshape(
                                2, nB, T + 1
                            )
                            cohorts["wx"][:, j, a] += np.bincount(
                                where, ww * deltaX[:, a], len(codes)
                            )

            def read(D, code):
                return Paths(D, T, code, cells, T_d, open_, linear_trends, continuous)

            for y, D, X, w, code, cl in batches():
                p = read(D, code)
                y = p.outcomes(y)
                X = np.where(p.present[:, :, None], X, np.nan)
                add_controls(p, y, X, w, cl)
                if not linear_trends:
                    add_switchers(p, y, X, w, eligibility(p, y), cl)
            if linear_trends:
                for y, D, X, w, code, cl in batches():
                    p = read(D, code)
                    y = p.outcomes(y)
                    X = np.where(p.present[:, :, None], X, np.nan)
                    add_switchers(p, y, X, w, eligibility(p, y), cl)
            codes, coh, coh_wx = cohorts["codes"], cohorts["coh"], cohorts["wx"]
            n_eff = min(effects, max(L_max[s] for s in directions))
            if n_eff < 1:
                raise ValueError(
                    "no effect can be estimated: no switcher has a group not yet switched to compare with"
                )
            n_pl = min(placebos, max(P_max[s] for s in directions) - linear_trends, n_eff)
            n_pl = max(n_pl, 0)
            theta = np.zeros((nL, Kc))
            if Kc:
                # Within each baseline, the outcome's one-period changes on the covariates'
                # among not-yet-switched rows, with period effects by cell
                # (did_multiplegt_main); and, for the variance's fitted values, with
                # period effects by baseline only (its feols call).
                with np.errstate(invalid="ignore", divide="ignore"):
                    mean_x = np.where(one_w[..., None] > 0, one_wx / one_w[..., None], 0.0)
                gram_x = (one_wxx - np.einsum("btk,btj->btkj", mean_x, one_wx)).sum(axis=1)
                cross_x = (one_wxy - mean_x * one_wy[..., None]).sum(axis=1)
                by_level = {}
                for name, array in (
                    ("n", one_n),
                    ("w", one_w),
                    ("wy", one_wy),
                    ("wx", one_wx),
                    ("wxy", one_wxy),
                    ("wxx", one_wxx),
                    ("gram", gram_x),
                    ("cross", cross_x),
                ):
                    by_level[name] = np.zeros((nL,) + array.shape[1:])
                    np.add.at(by_level[name], level, array)
                inv_denom = np.zeros((nL, Kc, Kc))
                theta_E = np.zeros((nL, Kc))
                for lv in range(nL):
                    Mb = by_level["gram"][lv]
                    if np.linalg.cond(Mb) > 1e12:
                        raise ValueError(
                            f"the covariates are collinear, or do not change, among the groups with "
                            f"first-period treatment {baselines[lv]:g} that have not switched yet"
                        )
                    invM = np.linalg.inv(Mb)
                    theta[lv] = invM @ by_level["cross"][lv]
                    inv_denom[lv] = invM * by_level["w"][lv].sum() * counted
                if trends_by is None:
                    theta_E = theta
                else:
                    Lw, Lwx = by_level["w"], by_level["wx"]
                    with np.errstate(invalid="ignore", divide="ignore"):
                        mean_L = np.where(Lw[..., None] > 0, Lwx / Lw[..., None], 0.0)
                    gram_L = (by_level["wxx"] - np.einsum("ltk,ltj->ltkj", mean_L, Lwx)).sum(axis=1)
                    cross_L = (by_level["wxy"] - mean_L * by_level["wy"][..., None]).sum(axis=1)
                    theta_E = np.stack(
                        [np.linalg.solve(gram_L[lv], cross_L[lv]) for lv in range(nL)]
                    )
                with np.errstate(invalid="ignore", divide="ignore"):
                    fixed = np.where(
                        by_level["w"] > 0,
                        (by_level["wy"] - np.einsum("ltk,lk->lt", by_level["wx"], theta_E))
                        / by_level["w"],
                        0.0,
                    )
                theta_cell = theta[level]
                ctrl_wy -= np.einsum("klbtj,bj->klbt", ctrl_wx, theta_cell)
                sw_wy -= np.einsum("jsbtk,bk->jsbt", sw_wx, theta_cell)
                if len(codes):
                    cb0 = coder.parts(codes)[0]
                    coh[:, :, 1] -= np.einsum("cjk,ck->cj", coh_wx, theta_cell[cb0])

            # Cells with controls; switchers in cells without any are not used.
            valid_cell = ctrl_w > 0
            if K == 2:
                valid_cell[1] &= ctrl_w[0] > 0
            slot_valid = valid_cell[slot_kind, slot_horizon]  # (S, nB, T + 1)
            sw_wx *= slot_valid[:, None, :, :, None]
            for s in (0, 1):
                if s not in directions:
                    sw_wx[:, s] = 0
            for arr in (sw_w, sw_wy, sw_n, sw_cum, sw_now):
                arr *= slot_valid[:, None]
                for s in (0, 1):
                    if s not in directions:
                        arr[:, s] = 0
            slot_ctrl_w = ctrl_w[slot_kind, slot_horizon]
            with np.errstate(invalid="ignore", divide="ignore"):
                ctrl_mean = np.where(ctrl_w > 0, ctrl_wy / ctrl_w, 0.0)
                t_any = sw_w.sum(axis=2) > 0  # (S, 2, T + 1)
                gate = t_any[:, :, None, :]
                un_w = sw_w + slot_ctrl_w[:, None] * gate
                union_n = sw_n + ctrl_n[slot_kind, slot_horizon][:, None] * gate
                union_mean = np.where(
                    un_w > 0, (sw_wy + ctrl_wy[slot_kind, slot_horizon][:, None] * gate) / un_w, 0.0
                )
            if n_cl:
                ctrl_n, union_n = _cluster_counts(
                    ctrl_pairs,
                    sw_pairs,
                    ctrl_n.shape,
                    union_n.shape,
                    n_clusters,
                    slots,
                    slot_valid,
                    directions,
                    gate,
                )
            # cohort tables, with cohorts in cells without controls removed
            cb, cF, cs = coder.parts(codes)
            cohort_n = coh[:, :, 2].T.copy()
            if n_cl:
                cohort_n[:] = 0
                for j, pairs_j in coh_pairs.items():
                    where_c = np.searchsorted(codes, pairs_j[:, 0])
                    cohort_n[j] = np.bincount(where_c, minlength=len(codes))
            cohort_w = coh[:, :, 0].T
            cohort_wy = coh[:, :, 1].T
            for j in range(S):
                at = cF + slot_horizon[j]
                ok = slot_valid[j, cb, np.minimum(at, T)] & (at <= T) & np.isin(cs, directions)
                cohort_n[j] *= ok
            with np.errstate(invalid="ignore", divide="ignore"):
                cohort_mean = np.where(cohort_w > 0, cohort_wy / cohort_w, 0.0)

            # The estimates: each a combination of slots by direction.
            Nk = sw_w.sum(axis=(2, 3))  # (S, 2)
            columns = []
            for kind, count in ((0, n_eff), (1, n_pl)):
                for i in range(1, count + 1):
                    if linear_trends:
                        members = [slot_index[(kind, h - 1, i)] for h in range(1, i + 1)]
                        target = slot_index[(kind, i - 1, i)]
                    else:
                        target = slot_index[(kind, i - 1, 0)]
                        members = [target]
                    columns.append((target, members))
            average = not linear_trends
            width = len(columns) + average
            coef = np.zeros((width, S, 2))
            for c, (target, members) in enumerate(columns):
                total = Nk[target].sum()
                if total <= 0:
                    continue
                delta = 0.0
                for s, sign in ((1, 1.0), (0, -1.0)):
                    share = Nk[target, s] / total
                    if share <= 0:
                        continue
                    # (as DIDmultiplegtDYN: the sum over horizons only if all have switchers)
                    used = members if all(Nk[m, s] > 0 for m in members) else [target]
                    for m in used:
                        coef[c, m, s] = share * sign / Nk[m, s]
                    delta += share * sw_cum[target, s].sum() / Nk[target, s]
                if normalized:
                    coef[c] /= delta
            weight_all = 0.0
            if average:
                # the average total effect: within each direction, the effects weighted by
                # their switchers over the average treatment change; then the directions
                # weighted by their switchers times that change (did_multiplegt_dyn)
                sums = {s: Nk[:n_eff, s].sum() for s in (0, 1)}
                den = {s: sw_now[:n_eff, s].sum() / sums[s] if sums[s] else 0.0 for s in (0, 1)}
                weight_in = den[1] * sums[1]
                weight_all = den[1] * sums[1] + den[0] * sums[0]
                share = {1: weight_in / weight_all if weight_all > 0 else 0.5}
                share[0] = 1 - share[1]
                for s, sign in ((1, 1.0), (0, -1.0)):
                    if sums[s] and den[s]:
                        coef[-1, :n_eff, s] = share[s] * sign / (sums[s] * den[s])
            if Kc:
                # M of did_multiplegt_dyn_core: the switchers' covariate changes less their
                # controls', per switcher; used where l <= T_d - 2; summed per baseline
                with np.errstate(invalid="ignore", divide="ignore"):
                    ratio = np.where(slot_ctrl_w[:, None] > 0, sw_w / slot_ctrl_w[:, None], 0.0)
                    moved = sw_wx.sum(axis=3) - np.einsum(
                        "jsbt,jbtk->jsbk", ratio, ctrl_wx[slot_kind, slot_horizon]
                    )
                    Mc = np.where(Nk[..., None, None] > 0, moved / Nk[..., None, None], 0.0)
                Mc *= (slot_horizon[:, None, None] + 1 <= T_d[None, None, :] - 2)[..., None]
                M_level = np.zeros((S, 2, nL, Kc))
                np.add.at(M_level, (slice(None), slice(None), level), Mc)
                constant = np.einsum("jslk,lk->js", M_level, theta)

            # Pass 3: each group's terms.
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
            cluster_sums = np.zeros((n_clusters, width)) if n_cl else None
            for y, D, X, w, code, cl in batches():
                p = read(D, code)
                y = p.outcomes(y)
                X = np.where(p.present[:, :, None], X, np.nan)
                kept = p.base >= 0
                eligible = eligibility(p, y)
                if Kc:
                    th = np.where(kept[:, None], theta_cell[np.maximum(p.base, 0)], 0.0)
                    y_raw, y = y, y - np.einsum("btk,bk->bt", X, th)
                cid = np.full(len(D), -1, dtype=np.int64)
                sw = p.up >= 0
                if sw.any() and len(codes):
                    code_sw = coder(p.base[sw], p.F[sw], p.switch_value[sw], p.up[sw])
                    pos = np.minimum(np.searchsorted(codes, code_sw), len(codes) - 1)
                    cid[sw] = np.where(codes[pos] == code_sw, pos, -1)
                U, V = group_terms(
                    p.base,
                    p.F,
                    p.up,
                    cid,
                    w,
                    y,
                    T_d,
                    only_never_switchers,
                    T,
                    slot_kind,
                    slot_horizon,
                    eligible,
                    *arrays,
                )
                if Kc:
                    b0 = np.maximum(p.base, 0)
                    lv = level[b0]
                    dY1, dX1 = _changes(y_raw, 1, 0), _changes(X, 1, 0)
                    rows1 = kept[:, None] & (tgrid >= 2) & (tgrid < p.F[:, None])
                    rows1 &= np.isfinite(dY1) & np.isfinite(dX1).all(axis=2)
                    n1 = by_level["n"][lv[:, None], tgrid]
                    factor = np.where(n1 >= 2, np.sqrt(n1 / np.maximum(n1 - 1, 1)), 1.0)
                    E = np.where(
                        n1 >= 2,
                        np.einsum("btk,bk->bt", dX1, theta_E[lv]) + fixed[lv[:, None], tgrid],
                        0.0,
                    )
                    N_c = by_level["w"][lv].sum(axis=1)[:, None]
                    term = np.where(rows1, w * factor * (dY1 - E) / N_c, 0.0)
                    resid = np.nan_to_num(dX1 - mean_x[b0[:, None], tgrid])
                    in_sum = np.einsum("btk,bt->bk", resid, term)
                    own = np.einsum("bkj,bj->bk", inv_denom[lv], in_sum) * (p.F >= 3)[:, None]
                    part2 = np.einsum("jsbk,bk->bjs", M_level[:, :, lv, :], own)
                    part2 = part2 - constant[None]
                    V -= np.where(kept[:, None, None], Nk[None] * part2 / counted, 0.0)
                estimate += np.einsum("bjs,cjs->c", U, coef)
                scores = np.einsum("bjs,cjs->bc", V, coef)
                if n_cl:
                    # the groups' terms are summed by cluster, as in DIDmultiplegtDYN
                    np.add.at(cluster_sums, cl, scores)
                else:
                    gram += scores.T @ scores
            if n_cl:
                gram = cluster_sums.T @ cluster_sums

            names = (
                [f"effect::{i}" for i in range(1, n_eff + 1)]
                + [f"placebo::{i}" for i in range(1, n_pl + 1)]
                + (["average_total_effect"] if average else [])
            )
            defined = np.array(
                [Nk[target].sum() > 0 for target, _ in columns]
                + ([weight_all > 0] if average else [])
            )
            estimate[~defined] = np.nan
            gram[~defined, :] = np.nan
            gram[:, ~defined] = np.nan
            targets = [target for target, _ in columns]
            # the store divides weights by their largest, then rescales them to average one
            scale = store.weight_normalizer / store.weight_scale
            table = pl.DataFrame(
                {
                    "coefficient": names[: len(columns)],
                    "switchers_in": [int(sw_n[j, 1].sum()) for j in targets],
                    "switchers_out": [int(sw_n[j, 0].sum()) for j in targets],
                    "weight_in": [float(sw_w[j, 1].sum()) / scale for j in targets],
                    "weight_out": [float(sw_w[j, 0].sum()) / scale for j in targets],
                }
            )
            labels = store.period_labels

            def cell_label(c):
                d = f"{c.real:g}"
                return d if trends_by is None else f"{d} | {trend_labels[int(c.imag)]}"

            diagnostics = dict(
                switchers=switchers,
                covariates=covariate_names,
                only_never_switchers=only_never_switchers,
                normalized=normalized,
                trends_by=trend_columns,
                linear_trends=linear_trends,
                weights=weights,
                groups_input=store.n_units_input,
                groups_used=groups_used,
                periods=T,
                time_min=labels.label_int(int(store.times[0])),
                time_max=labels.label_int(int(store.times[-1])),
                baselines=baselines.tolist(),
                last_period_with_controls={
                    cell_label(c): labels.label_int(int(store.times[t - 1])) if t >= 1 else None
                    for c, t in zip(cells.tolist(), T_d.tolist(), strict=True)
                },
                cluster=cluster,
                continuous_baseline=degree,
                inference=f"clustered by {cluster or 'group'}; the authors' conservative "
                "variance (cohort-demeaned)",
                batch_units=B,
                partitions=store.partition_count,
            )
            if continuous:
                message = (
                    "With a continuous first-period treatment, the authors note that the "
                    "analytic standard errors are not backed by a proven asymptotic result: "
                    "conservative with a correctly specified polynomial in simulations, they "
                    "can be too small if it is misspecified (DIDmultiplegtDYN's continuous "
                    "option). Check the results' sensitivity to continuous_baseline, and "
                    "consider refit_bootstrap(estimator=did_dcdh), as the authors recommend."
                )
                diagnostics["warnings"] = [message]
                warnings.warn(message, ContinuousBaselineWarning, 3)
            return DCDHResult(
                fml=fml,
                depvar=yname,
                coefnames=names,
                beta=estimate,
                vcov=gram,
                vcov_type="CRV0",
                df_t=(n_clusters or N) - 1,
                all_vcovs={"CRV0": (gram, (n_clusters or N) - 1)},
                n_obs=store.n_rows_used,
                n_clusters={cluster: n_clusters} if n_cl else {panel[0]: N},
                n_units=N,
                diagnostics=diagnostics,
                _switchers=table,
                _internal=dict(diagnostics),
            )


def _baseline_polynomial(D, degree):
    """1{t >= j} d^k for periods j = 2..T and k = 1..degree, with d the group's
    treatment in its first observed period (DIDmultiplegtDYN's time_fe_XX_j_btk_XX)."""
    n, T = D.shape
    observed = np.isfinite(D)
    d = D[np.arange(n), observed.argmax(axis=1)]
    t = np.arange(1, T + 1)
    columns = [
        np.where(observed, (t[None, :] >= j) * d[:, None] ** k, np.nan)
        for j in range(2, T + 1)
        for k in range(1, degree + 1)
    ]
    return np.stack(columns, axis=2) if columns else np.zeros((n, T, 0))


def _group_cluster(values):
    """Each group's cluster index, which must not change over its observed periods."""
    high, low = np.fmax.reduce(values, axis=1), np.fmin.reduce(values, axis=1)
    if np.any(high != low):
        raise ValueError(
            "each group must be in one cluster: cluster must be constant within groups"
        )
    return high.astype(np.int64)


def _cluster_counts(
    ctrl_pairs, sw_pairs, ctrl_shape, union_shape, C, slots, valid, directions, gate
):
    """Distinct clusters among each cell-period's controls, and among the union of a
    slot's controls and switchers in each direction (DIDmultiplegtDYN's cluster counts).
    """
    K, Lm, nB, T1 = ctrl_shape
    ctrl = np.zeros(ctrl_shape)
    for (k, li), keys in ctrl_pairs.items():
        ctrl[k, li] = np.bincount(keys // C, minlength=nB * T1).reshape(nB, T1)
    union = np.zeros(union_shape)
    for j, (k, li, _) in enumerate(slots):
        controls = ctrl_pairs.get((k, li), np.zeros(0, dtype=np.int64))
        c_cell, c_cluster = controls // C, controls % C
        switch = sw_pairs.get(j, np.zeros(0, dtype=np.int64))
        s_flat, s_cluster = switch // C, switch % C
        s_dir, s_cell = s_flat // (nB * T1), s_flat % (nB * T1)
        for s in directions:
            mine = (s_dir == s) & valid[j].ravel()[s_cell]
            open_t = gate[j, s, 0].ravel() if gate.ndim == 4 else gate[j, s]
            ctrl_on = open_t[c_cell % T1]
            keys = np.unique(
                np.r_[s_cell[mine] * C + s_cluster[mine], c_cell[ctrl_on] * C + c_cluster[ctrl_on]]
            )
            union[j, s] = np.bincount(keys // C, minlength=nB * T1).reshape(nB, T1)
    return ctrl, union


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
    trends_by=None,
    linear_trends=False,
    weights=None,
    allow_time_varying_weights=False,
    incomplete_units="error",
    cluster=None,
    continuous_baseline=None,
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
    the authors' R and Stata ``did_multiplegt_dyn``. Weights may vary over time (with
    ``allow_time_varying_weights=True``): a group enters each comparison with its weight
    in the period the effect is measured at (for a placebo, that of the matching effect).

    de Chaisemartin, C. and D'Haultfoeuille, X. (2024). Difference-in-differences
    estimators of intertemporal treatment effects. NBER Working Paper 29873.

    Parameters
    ----------
    fml : str
        ``"y ~ 1"``, or ``"y ~ x + z"`` with covariates: within each first-period
        treatment, the outcome's changes among groups not yet switched are regressed on
        the covariates' changes with period effects, and the comparisons use the
        outcome changes net of the covariates' fitted part.
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
    trends_by : str or list of str, optional
        Group-level columns (constant within each group) within which trends may
        differ: switchers are compared only with groups that have the same values of
        these columns as well as the same first-period treatment.
    linear_trends : bool, default False
        Allow each group its own linear trend. The comparisons use the first
        differences of the outcome (and of the covariates), and effect l is the sum of
        the first l such effects, for the switchers that have all l; placebo l
        likewise. Groups that first switch in the second period are not used, and the
        average total effect is not estimated.
    {WEIGHTS}
    incomplete_units : {"error", "drop", "keep"}, default "error"
        What to do with groups missing some periods: refuse the data, drop those
        groups, or keep their observed periods. With "keep", a change is used only where
        both of its periods are observed, and a group not observed just before its
        switch is used only as a control until its last observed period before it, as
        in DIDmultiplegtDYN.
    cluster : str, optional
        A column to cluster the standard errors by, above the group (each group in
        one cluster), as ``cluster`` in DIDmultiplegtDYN: the groups' terms are summed
        by cluster, and the authors' cohort demeaning counts clusters rather than
        groups. Default: the group.
    continuous_baseline : int, optional
        For a first-period treatment that is continuous, so that no two groups share
        one (DIDmultiplegtDYN's ``continuous``): the degree of a polynomial in it.
        All groups are then compared together, with untreated trends that may differ
        by a polynomial of this degree in the first-period treatment, period by
        period (entered as covariates), and the switching cohorts of the variance
        are defined by the direction of the switch. The standard errors are not
        backed by a proven asymptotic result (see the documentation); fits warn.
    {RESOURCES}

    Returns
    -------
    DCDHResult or Profile
        ``effect::l``, ``placebo::l`` and ``average_total_effect`` (not with
        ``linear_trends``) with their joint covariance; ``pretrend_test()``, ``effects_equal_test()``, ``switchers()``.
    """
    options = dict(locals())
    fml, data = options.pop("fml"), options.pop("data")
    return public_profile(_dcdh(fml, data, **_options.internal(options)))
