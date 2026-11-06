"""R did-compatible observed-row CS functionals, clustered by original unit.

Each date pair uses repeated-cross-section mean functionals on its observed
rows. Units seen at only one date contribute; scores are folded by unit.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from ._core.functionals import NormalizedMeans
from ._core.nuisance import fit_linear_systems, fit_systems, solve_checked
from .cohort_scores import CohortScoreCovariance


@dataclass
class RowComparison:
    cohort: int
    controls: tuple
    target: int
    base: int

    def weights(self, raw, unit, post):
        w = raw[3]
        return w[unit, np.where(post, self.target, self.base)] if w.ndim == 2 else w[unit]

    def select(self, raw):
        g, y, x = raw[:3]
        mask = (g == self.cohort) | np.isin(g, self.controls)
        dates = np.array([self.base, self.target])
        unit, period = np.nonzero(mask[:, None] & np.isfinite(y[:, dates]))
        ti = dates[period]
        return unit, x[unit, ti], g[unit] == self.cohort, y[unit, ti], period == 1

    def __call__(self, raw):
        unit, X, d, y, post = self.select(raw)
        return (
            (X, d, y[:, None], self.weights(raw, unit, post))
            if len(raw) == 4
            else (X, d, y[:, None])
        )


class RowFunctional:
    def __init__(self, comparison, method, trim, system, models, width):
        self.c, self.method, self.trim, self.system = comparison, method, trim, system
        # Model order is control pre/post, treated pre/post.
        self.models = models
        if method == "ipw":
            self.signs = np.array([1, -1, -1, 1.0])
            self.value_derivative = np.zeros((4, 0))
        elif method == "reg":
            self.signs = np.array([1, -1, -1.0])
            self.value_derivative = np.array([[0, 0], [0, 0], [-1, 1.0]])
        else:
            self.signs = np.array([1, -1, -1, 1, 1, -1, -1, 1.0])
            self.value_derivative = np.array(
                [
                    [0, -1, 0, 0],
                    [-1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [-1, 0, 0, 0],
                    [0, -1, 0, 1],
                    [0, -1, 0, 1],
                    [-1, 0, 1, 0],
                    [-1, 0, 1, 0.0],
                ]
            )
        self.means = NormalizedMeans(len(self.signs), width)
        self.H = np.zeros((width, width))
        self.trimmed = 0
        self.capped = 0
        self.stratum_counts = np.zeros(4, dtype=int)
        self.probability_min, self.probability_max = np.inf, -np.inf
        self.control_weight_squares = np.zeros(2)
        self.sample_weight = np.zeros(4)
        self.sample_weight_squares = np.zeros(4)

    def rows(self, raw):
        unit, X, d, y, post = self.c.select(raw)
        tp, tm = d & post, d & ~post
        cp, cm = ~d & post, ~d & ~post
        p = np.minimum(expit(X @ self.system.beta), 1 - 1e-6) if self.system is not None else None
        odds = (p / (1 - p)) * (p < self.trim) if p is not None else np.ones(len(d))
        if self.method == "ipw":
            weights = np.column_stack([tp, tm, cp * odds, cm * odds])
            values = np.broadcast_to(y[:, None], weights.shape)
        elif self.method == "reg":
            weights = np.column_stack([tp, tm, d]).astype(float)
            values = np.column_stack([y, y, X @ (self.models[1][0] - self.models[0][0])])
        else:
            m = [X @ b for b, _ in self.models]
            r = y - np.where(post, m[1], m[0])
            weights = np.column_stack([tp, tm, cp * odds, cm * odds, d, tp, d, tm])
            values = np.column_stack(
                [r, r, r, r, m[3] - m[1], m[3] - m[1], m[2] - m[0], m[2] - m[0]]
            )
        if len(raw) == 4:
            weights = weights * self.c.weights(raw, unit, post)[:, None]
        return unit, X, d, y, post, p, weights, values

    def update(self, raw):
        unit, X, d, _, post, p, w, v = self.rows(raw)
        sw = self.c.weights(raw, unit, post) if len(raw) == 4 else np.ones(len(d))
        for j, mask in enumerate((~d & ~post, ~d & post, d & ~post, d & post)):
            self.sample_weight[j] += sw[mask].sum()
            self.sample_weight_squares[j] += sw[mask] @ sw[mask]
        self.means.update(w, v, X)
        self.stratum_counts += [
            np.sum(~d & ~post),
            np.sum(~d & post),
            np.sum(d & ~post),
            np.sum(d & post),
        ]
        if p is not None:
            self.H += X.T @ ((sw * p * (1 - p))[:, None] * X)
            self.trimmed += int(((~d) & (p >= self.trim)).sum())
            self.capped += int((expit(X @ self.system.beta) > 1 - 1e-6).sum())
            self.control_weight_squares += (w[:, [3, 2]] ** 2).sum(axis=0)
            if len(p):
                self.probability_min = min(self.probability_min, float(p.min()))
                self.probability_max = max(self.probability_max, float(p.max()))

    def finish(self):
        self.means.finish()
        self.estimate = self.means.mean @ self.signs
        J = (self.value_derivative.T * self.signs) @ self.means.mean_x
        self.reg_correction = [
            solve_checked(Q, j, "outcome score correction")
            for (_, Q), j in zip(self.models, J, strict=True)
        ]
        if self.system is not None:
            Jps = -self.means.centered_x[2] + self.means.centered_x[3]
            self.ps_correction = solve_checked(self.H, Jps, "propensity score correction")

    def scores(self, raw):
        unit, X, d, y, post, p, w, v = self.rows(raw)
        phi = self.means.scores(w, v, self.signs)
        sw = self.c.weights(raw, unit, post) if len(raw) == 4 else np.ones(len(d))
        strata = [~d & ~post, ~d & post, d & ~post, d & post]
        for (b, _), correction, mask in zip(
            self.models, self.reg_correction, strata[: len(self.models)], strict=True
        ):
            phi += sw * mask * (y - X @ b) * (X @ correction)
        if self.system is not None:
            phi += sw * (d - p) * (X @ self.ps_correction)
        return unit, phi


def fit_cs_unbalanced(
    store, groups, cells, bases, controls, method, trim, multiplier=None, *, balance_collector=None
):
    K = len(store.design.names) if store.design is not None else 1
    names = store.design.names if store.design is not None else ["Intercept"]
    comparisons = [
        RowComparison(g, tuple(np.asarray(groups)[c]), t - store.times[0], int(b))
        for (g, t), b, c in zip(cells, bases, controls, strict=True)
    ]

    def batches():
        return store.batches(
            include_x=True, include_weights=True, row_weights=not store.weights_time_invariant
        )

    linear_keys = {}
    model_indices = []
    for c in comparisons:
        keys = [] if method == "ipw" else [(c.controls, c.base), (c.controls, c.target)]
        if method == "dr":
            keys += [((c.cohort,), c.base), ((c.cohort,), c.target)]
        model_indices.append([linear_keys.setdefault(key, len(linear_keys)) for key in keys])

    def linear_transform(cohorts, ti):
        def select(raw):
            g, y, x, w = raw
            mask = np.isin(g, cohorts) & np.isfinite(y[:, ti])
            return x[mask, ti], y[mask, ti], w[mask, ti] if w.ndim == 2 else w[mask]

        return select

    models = fit_linear_systems(batches, [linear_transform(*key) for key in linear_keys], K)
    ps_keys, ps_indices, ps_transforms = {}, [], []
    for c in comparisons:
        key = (c.cohort, c.controls, *sorted([c.base, c.target]))
        if key not in ps_keys:
            ps_keys[key] = len(ps_keys)
            ps_transforms.append(c)
        ps_indices.append(ps_keys[key])
    systems, passes = (
        fit_systems(batches, ps_transforms, "ipw", names) if method != "reg" else ([], 0)
    )
    fitted = [
        RowFunctional(c, method, trim, systems[ps] if systems else None, [models[k] for k in ix], K)
        for c, ps, ix in zip(comparisons, ps_indices, model_indices, strict=True)
    ]
    for raw in batches():
        for f in fitted:
            f.update(raw)
    for f in fitted:
        f.finish()
    M = len(cells)
    cohorts = np.array([g for g, _ in cells])
    counts = np.array([store.gw[g] for g, _ in cells], float)
    # Unadjusted sums of [scores, share] outer products: api.py owns the
    # explicit CRV0/CRV1 factor for all covariance blocks.
    accumulator = CohortScoreCovariance(cohorts, counts / store.n_units)
    for raw in batches():
        g, y, _, w = raw
        scores = np.zeros((len(g), M))
        anchor = w.sum(axis=1) / np.isfinite(y).sum(axis=1) if w.ndim == 2 else w
        for j, f in enumerate(fitted):
            unit, phi = f.scores(raw)
            np.add.at(scores[:, j], unit, phi)
        accumulator.add(g, scores, anchor)
        if multiplier is not None:
            share = anchor[:, None] * ((g[:, None] == cohorts) - counts / store.n_units)
            multiplier.update(scores, share)
    V, cross, share_cov = accumulator.blocks()
    if balance_collector is not None:
        balance_collector.unbalanced(comparisons, fitted, batches)
    metadata = dict(
        est_method=method,
        covariate_names=names,
        covariate_convention="observed-date covariates; repeated-cross-section functionals",
        nuisance_systems=len(linear_keys),
        propensity_models=len(systems),
        nuisance_source_passes=passes + 2 + int(method != "ipw"),
        trim_level=trim,
        comparisons=[
            dict(
                cohort=c.cohort,
                time=int(store.times[c.target]),
                base=int(store.times[c.base]),
                trimmed_controls=f.trimmed,
                capped_probabilities=f.capped,
                n_control_pre=int(f.stratum_counts[0]),
                n_control_post=int(f.stratum_counts[1]),
                n_treated_pre=int(f.stratum_counts[2]),
                n_treated_post=int(f.stratum_counts[3]),
                propensity_range=[f.probability_min, f.probability_max]
                if method != "reg"
                else None,
                effective_controls=(f.means.weight[[3, 2]] ** 2 / f.control_weight_squares).tolist()
                if method != "reg"
                else (f.sample_weight[:2] ** 2 / f.sample_weight_squares[:2]).tolist(),
                effective_treated=(f.sample_weight[2:] ** 2 / f.sample_weight_squares[2:]).tolist(),
                logit_iterations=f.system.iterations if method != "reg" else None,
            )
            for c, f in zip(comparisons, fitted, strict=True)
        ],
        warnings=[],
    )
    if any(f.trimmed for f in fitted):
        metadata["warnings"].append(
            "Propensity trimming removed control contributions; inspect comparisons."
        )
    if any(f.capped for f in fitted):
        metadata["warnings"].append(
            "Some propensity probabilities were capped at 1-1e-6 under the R DRDID convention."
        )
    return (
        np.array([f.estimate for f in fitted]),
        counts,
        V,
        cross,
        share_cov,
        metadata,
    )
