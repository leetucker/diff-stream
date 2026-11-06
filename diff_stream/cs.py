"""CS comparison populations and earlier-period covariate conventions."""

from dataclasses import dataclass

import numpy as np
from hdfe_stream.core.influence import ClusterScoreAccumulator

from ._core.contrasts import fit_contrasts
from ._core.nuisance import fit_systems


@dataclass
class Comparison:
    cohort: int
    controls: tuple
    xperiod: int
    cells: list
    targets: np.ndarray
    bases: np.ndarray
    weightperiod: int = 0

    def weights(self, raw, mask):
        w = raw[3]
        return w[mask, self.weightperiod] if w.ndim == 2 else w[mask]

    def select(self, raw):
        g, y, x = raw[:3]
        mask = (g == self.cohort) | np.isin(g, self.controls)
        return (
            mask,
            x[mask, self.xperiod, :],
            g[mask] == self.cohort,
            y[mask][:, self.targets] - y[mask][:, self.bases],
        )

    def __call__(self, raw):
        mask, X, d, Y = self.select(raw)
        return (X, d, Y, self.weights(raw, mask)) if len(raw) == 4 else (X, d, Y)


def fit_cs(
    store,
    groups,
    cells,
    bases,
    controls,
    method,
    trim_level,
    multiplier=None,
    *,
    balance_collector=None,
):
    """Fit cached comparison systems, then assemble cross-cell unit covariance."""
    comparisons = {}
    for j, (g, t) in enumerate(cells):
        controls_g = tuple(groups[k] for k in np.flatnonzero(controls[j]))
        tindex = t - store.times[0]
        xb = 0 if store.x_time_invariant else min(tindex, int(bases[j]))
        wp = min(tindex, int(bases[j])) if not store.weights_time_invariant else 0
        key = (g, controls_g, xb, wp)
        if key not in comparisons:
            comparisons[key] = [[], [], []]
        comparisons[key][0].append(j)
        comparisons[key][1].append(tindex)
        comparisons[key][2].append(bases[j])
    transforms = [
        Comparison(g, c, xb, ix, np.array(t, dtype=int), np.array(b, dtype=int), wp)
        for (g, c, xb, wp), (ix, t, b) in comparisons.items()
    ]

    def batches():
        return store.batches(
            include_x=True, include_weights=True, row_weights=not store.weights_time_invariant
        )

    names = store.design.names if store.design is not None else ["Intercept"]
    systems, passes = fit_systems(batches, transforms, method, names)
    fitted = fit_contrasts(batches, transforms, systems, method, trim_level)
    M = len(cells)
    beta = np.zeros(M)
    for c, f in zip(transforms, fitted, strict=True):
        beta[c.cells] = f.estimate
    accumulator = ClusterScoreAccumulator()
    cohorts = np.array([g for g, t in cells])
    for raw in batches():
        g, _, _, w = raw
        scores = np.zeros((len(g), M))
        counts = np.array([store.gw[g] for g, t in cells], dtype=float)
        anchor = w[:, 0] if w.ndim == 2 else w
        share = anchor[:, None] * ((g[:, None] == cohorts[None, :]) - counts / store.n_units)
        for c, f in zip(transforms, fitted, strict=True):
            mask, X, d, Y = c.select(raw)
            scores[np.ix_(np.flatnonzero(mask), c.cells)] = f.scores(X, d, Y, c.weights(raw, mask))
        accumulator.add_clusters(scores, share)
        if multiplier is not None:
            multiplier.update(scores, share)
    counts = np.array([store.gw[g] for g, t in cells], dtype=float)
    # add_clusters consumes per-unit scores and sums does not apply an SSC.
    # api.py owns the explicit CRV0/CRV1 factor for all covariance blocks.
    joint = accumulator.sums
    V, cross, share_cov = joint[:M, :M], joint[:M, M:], joint[M:, M:]
    if balance_collector is not None:
        balance_collector.balanced(transforms, fitted, cells, batches)
    metadata = dict(
        est_method=method,
        covariate_names=names,
        covariates_time_invariant=store.x_time_invariant,
        nuisance_systems=len(transforms),
        propensity_models=0 if method == "reg" else len(transforms),
        nuisance_source_passes=passes + 2,
        trim_level=trim_level,
        probability_cap=1 - 1e-6 if method != "reg" else None,
        comparisons=[
            dict(
                cohort=c.cohort,
                controls=list(c.controls),
                covariate_time=int(store.times[c.xperiod]),
                weight_time=int(store.times[c.weightperiod])
                if not store.weights_time_invariant
                else None,
                cells=c.cells,
                **f.diagnostics,
            )
            for c, f in zip(transforms, fitted, strict=True)
        ],
    )
    metadata["warnings"] = []
    if any(f.diagnostics["trimmed_controls"] for f in fitted):
        metadata["warnings"].append(
            "Propensity trimming removed control contributions in some comparisons; inspect diagnostics['comparisons']."
        )
    if any(f.diagnostics["capped_probabilities"] for f in fitted):
        metadata["warnings"].append(
            "Some propensity probabilities were capped at 1-1e-6 under the R DRDID convention."
        )
    return beta, counts, V, cross, share_cov, metadata
